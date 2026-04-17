import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class VAConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    vision_variant: str = "So400m/14"
    decoder_width: int = 512
    decoder_depth: int = 4
    decoder_num_heads: int = 8
    decoder_mlp_dim: int = 2048
    temperatures: tuple[float, ...] = (0.02, 0.05, 0.2)

    action_dim: int = 32
    action_horizon: int = 16
    max_token_len: int = 1

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.VA

    @override
    def create(self, rng: at.KeyArrayLike) -> "VA":
        from openpi.models.va import VA

        return VA(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation_spec, action_spec
