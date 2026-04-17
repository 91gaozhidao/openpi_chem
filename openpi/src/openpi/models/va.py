import dataclasses

import einops
import flax.linen as nn
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import dbp_loss
from openpi.models import model as _model
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

if False:  # pragma: no cover
    from openpi.models.va_config import VAConfig


class _DecoderMLP(nn.Module):
    width: int
    mlp_dim: int
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.mlp_dim, dtype=self.dtype_mm)(x)
        x = nn.gelu(x)
        return nn.Dense(self.width, dtype=self.dtype_mm)(x)


class _DecoderBlock(nn.Module):
    width: int
    num_heads: int
    mlp_dim: int
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, queries, context, self_mask, cross_mask):
        residual = queries
        queries = nn.LayerNorm(dtype=self.dtype_mm)(queries)
        queries = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=self.dtype_mm,
            kernel_init=nn.initializers.xavier_uniform(),
        )(queries, queries, mask=self_mask)
        queries = residual + queries

        residual = queries
        queries = nn.LayerNorm(dtype=self.dtype_mm)(queries)
        queries = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=self.dtype_mm,
            kernel_init=nn.initializers.xavier_uniform(),
        )(queries, context, mask=cross_mask)
        queries = residual + queries

        residual = queries
        queries = nn.LayerNorm(dtype=self.dtype_mm)(queries)
        queries = residual + _DecoderMLP(self.width, self.mlp_dim, dtype_mm=self.dtype_mm)(queries)
        return queries


class _ActionDecoder(nn.Module):
    action_horizon: int
    width: int
    depth: int
    num_heads: int
    mlp_dim: int
    action_dim: int
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, context, context_mask, noise_tokens):
        batch_size = context.shape[0]
        position_embeddings = self.param(
            "position_embeddings",
            nn.initializers.normal(stddev=0.02),
            (1, self.action_horizon, self.width),
            jnp.float32,
        )
        queries = (noise_tokens + position_embeddings).astype(self.dtype_mm)

        causal_mask = jnp.tril(jnp.ones((self.action_horizon, self.action_horizon), dtype=jnp.bool_))
        self_mask = jnp.broadcast_to(
            causal_mask[None, None, :, :],
            (batch_size, 1, self.action_horizon, self.action_horizon),
        )
        cross_mask = jnp.broadcast_to(
            context_mask[:, None, None, :],
            (batch_size, 1, self.action_horizon, context.shape[1]),
        )

        for layer_idx in range(self.depth):
            queries = _DecoderBlock(
                self.width,
                self.num_heads,
                self.mlp_dim,
                dtype_mm=self.dtype_mm,
                name=f"decoder_block_{layer_idx}",
            )(queries, context, self_mask, cross_mask)

        queries = nn.LayerNorm(dtype=self.dtype_mm, name="decoder_norm")(queries)
        return nn.Dense(self.action_dim, dtype=jnp.float32, name="action_head")(queries)


def _repeat_batch(x: at.Array | None, repeats: int) -> at.Array | None:
    if x is None:
        return None
    return jnp.repeat(x, repeats, axis=0)


def _repeat_observation(observation: _model.Observation, repeats: int) -> _model.Observation:
    return _model.Observation(
        images={k: _repeat_batch(v, repeats) for k, v in observation.images.items()},
        image_masks={k: _repeat_batch(v, repeats) for k, v in observation.image_masks.items()},
        state=_repeat_batch(observation.state, repeats),
        tokenized_prompt=_repeat_batch(observation.tokenized_prompt, repeats),
        tokenized_prompt_mask=_repeat_batch(observation.tokenized_prompt_mask, repeats),
        token_ar_mask=_repeat_batch(observation.token_ar_mask, repeats),
        token_loss_mask=_repeat_batch(observation.token_loss_mask, repeats),
    )


class VA(_model.BaseModel):
    def __init__(self, config: "VAConfig", rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config
        vision_width = _siglip.decode_variant(config.vision_variant)["width"]

        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=None,
                variant=config.vision_variant,
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(img=img)

        self.vision_proj = nnx.Linear(vision_width, config.decoder_width, rngs=rngs)
        self.state_proj = nnx.Linear(config.action_dim, config.decoder_width, rngs=rngs)
        self.noise_proj = nnx.Linear(config.action_dim, config.decoder_width, rngs=rngs)

        decoder = nnx_bridge.ToNNX(
            _ActionDecoder(
                action_horizon=config.action_horizon,
                width=config.decoder_width,
                depth=config.decoder_depth,
                num_heads=config.decoder_num_heads,
                mlp_dim=config.decoder_mlp_dim,
                action_dim=config.action_dim,
                dtype_mm=config.dtype,
            )
        )
        decoder.lazy_init(
            jnp.ones((1, 2, config.decoder_width), dtype=jnp.float32),
            jnp.ones((1, 2), dtype=jnp.bool_),
            jnp.ones((1, config.action_horizon, config.decoder_width), dtype=jnp.float32),
            rngs=rngs,
        )
        self.decoder = decoder

        # This gets toggled by model.train() / model.eval().
        self.deterministic = True

    @at.typecheck
    def encode_context(
        self,
        observation: _model.Observation,
        *,
        train: bool,
    ) -> tuple[at.Float[at.Array, "b t d"], at.Bool[at.Array, "b t"]]:
        tokens = []
        masks = []
        for name in observation.images:
            image_tokens, _ = self.PaliGemma.img(observation.images[name], train=train)
            image_tokens = self.vision_proj(image_tokens)
            tokens.append(image_tokens)
            masks.append(einops.repeat(observation.image_masks[name], "b -> b s", s=image_tokens.shape[1]))

        state_token = self.state_proj(observation.state)[:, None, :]
        tokens.append(state_token)
        masks.append(jnp.ones((observation.state.shape[0], 1), dtype=jnp.bool_))

        return jnp.concatenate(tokens, axis=1), jnp.concatenate(masks, axis=1)

    @at.typecheck
    def predict_actions(
        self,
        observation: _model.Observation,
        *,
        train: bool,
        noise: _model.Actions | None = None,
    ) -> _model.Actions:
        if noise is None:
            raise ValueError("VA.predict_actions requires an explicit noise tensor.")
        context_tokens, context_mask = self.encode_context(observation, train=train)
        noise_tokens = self.noise_proj(noise)
        return self.decoder(context_tokens, context_mask, noise_tokens)

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, " b"]:
        preprocess_rng, noise_rng = jax.random.split(rng)
        observation = _model.preprocess_observation(
            preprocess_rng,
            observation,
            train=train,
            image_keys=list(observation.images.keys()),
        )

        batch_size = actions.shape[0]
        gen_samples = self.config.gen_samples
        action_noise = jax.random.normal(
            noise_rng,
            (batch_size, gen_samples, self.action_horizon, self.action_dim),
            dtype=actions.dtype,
        )
        repeated_observation = _repeat_observation(observation, gen_samples)
        pred_actions = self.predict_actions(
            repeated_observation,
            train=train,
            noise=action_noise.reshape(batch_size * gen_samples, self.action_horizon, self.action_dim),
        )
        loss, _ = dbp_loss.compute_dbp_loss(
            preds=pred_actions.reshape(batch_size, gen_samples, -1),
            pos_targets=actions.reshape(actions.shape[0], 1, -1),
            temp_schedule=self.config.temperatures,
        )
        return loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        num_steps: int | at.Int[at.Array, ""] | None = None,
    ) -> _model.Actions:
        del num_steps
        observation = _model.preprocess_observation(
            None,
            observation,
            train=False,
            image_keys=list(observation.images.keys()),
        )
        if noise is None:
            noise = jax.random.normal(rng, (observation.state.shape[0], self.action_horizon, self.action_dim))
        return self.predict_actions(observation, train=False, noise=noise)
