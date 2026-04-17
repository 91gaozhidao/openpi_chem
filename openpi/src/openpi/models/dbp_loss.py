import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at


@at.typecheck
def compute_pairwise_euclidean_distance(
    x: at.Float[at.Array, "b n d"],
    y: at.Float[at.Array, "b m d"],
    eps: float = 1e-8,
) -> at.Float[at.Array, "b n m"]:
    """Compute pairwise Euclidean distance with numerically stable sqrt."""
    xy_dot_product = jnp.einsum("bnd,bmd->bnm", x, y, preferred_element_type=jnp.float32)
    x_squared_norms = jnp.einsum("bnd,bnd->bn", x, x, preferred_element_type=jnp.float32)
    y_squared_norms = jnp.einsum("bmd,bmd->bm", y, y, preferred_element_type=jnp.float32)
    squared_distance = x_squared_norms[:, :, None] + y_squared_norms[:, None, :] - 2 * xy_dot_product
    return jnp.sqrt(jnp.maximum(squared_distance, eps))


@at.typecheck
def compute_dbp_loss(
    preds: at.Float[at.Array, "b n d"],
    pos_targets: at.Float[at.Array, "b p d"],
    neg_targets: at.Float[at.Array, "b q d"] | None = None,
    w_pred: at.Float[at.Array, "b n"] | None = None,
    w_pos: at.Float[at.Array, "b p"] | None = None,
    w_neg: at.Float[at.Array, "b q"] | None = None,
    temp_schedule: tuple[float, ...] = (0.02, 0.05, 0.2),
) -> tuple[at.Float[at.Array, " b"], dict[str, at.Array]]:
    """JAX implementation of the DBP loss used for one-step VA training."""
    batch_size, num_preds, seq_len = preds.shape
    num_pos = pos_targets.shape[1]

    preds = preds.astype(jnp.float32)
    pos_targets = pos_targets.astype(jnp.float32)

    if neg_targets is None:
        neg_targets = jnp.zeros((batch_size, 0, seq_len), dtype=jnp.float32)
    else:
        neg_targets = neg_targets.astype(jnp.float32)
    num_neg = neg_targets.shape[1]

    if w_pred is None:
        w_pred = jnp.ones((batch_size, num_preds), dtype=jnp.float32)
    else:
        w_pred = w_pred.astype(jnp.float32)
    if w_pos is None:
        w_pos = jnp.ones((batch_size, num_pos), dtype=jnp.float32)
    else:
        w_pos = w_pos.astype(jnp.float32)
    if w_neg is None:
        w_neg = jnp.ones((batch_size, num_neg), dtype=jnp.float32)
    else:
        w_neg = w_neg.astype(jnp.float32)

    anchored_preds = jax.lax.stop_gradient(preds)
    all_targets = jnp.concatenate([anchored_preds, neg_targets, pos_targets], axis=1)
    all_weights = jnp.concatenate([w_pred, w_neg, w_pos], axis=1)

    diagnostics: dict[str, at.Array] = {}

    dists = compute_pairwise_euclidean_distance(anchored_preds, all_targets)
    weighted_dists = dists * all_weights[:, None, :]
    scale = weighted_dists.mean() / jnp.maximum(all_weights.mean(), 1e-8)
    diagnostics["global_scale"] = scale

    dim_scale = jnp.maximum(scale / jnp.sqrt(float(seq_len)), 1e-3)
    anchored_preds_norm = anchored_preds / dim_scale
    targets_norm = all_targets / dim_scale
    dists_norm = dists / jnp.maximum(scale, 1e-3)

    identity_mask = jnp.eye(num_preds, dtype=jnp.float32)
    spatial_block_mask = jnp.pad(identity_mask, ((0, 0), (0, num_neg + num_pos)))[None, :, :]
    dists_norm = dists_norm + spatial_block_mask * 100.0

    aggregated_forces = jnp.zeros_like(anchored_preds_norm)
    split_boundary = num_preds + num_neg

    for temperature in temp_schedule:
        logits = -dists_norm / temperature
        affinity_forward = jax.nn.softmax(logits, axis=-1)
        affinity_backward = jax.nn.softmax(logits, axis=-2)
        mutual_affinity = jnp.sqrt(jnp.maximum(affinity_forward * affinity_backward, 1e-6))
        mutual_affinity = mutual_affinity * all_weights[:, None, :]

        affinity_neg_cluster = mutual_affinity[:, :, :split_boundary]
        affinity_pos_cluster = mutual_affinity[:, :, split_boundary:]

        sum_pos_attraction = affinity_pos_cluster.sum(axis=-1, keepdims=True)
        repulsive_coeff = -affinity_neg_cluster * sum_pos_attraction

        sum_neg_repulsion = affinity_neg_cluster.sum(axis=-1, keepdims=True)
        attractive_coeff = affinity_pos_cluster * sum_neg_repulsion

        force_coeffs = jnp.concatenate([repulsive_coeff, attractive_coeff], axis=2)
        total_gradient_force = jnp.einsum("biy,byx->bix", force_coeffs, targets_norm, preferred_element_type=jnp.float32)
        accumulated_coeffs = force_coeffs.sum(axis=-1)
        total_gradient_force = total_gradient_force - accumulated_coeffs[:, :, None] * anchored_preds_norm

        force_magnitude = jnp.mean(jnp.square(total_gradient_force))
        diagnostics[f"force_magnitude_T{temperature:.2f}"] = force_magnitude
        force_scale = jnp.sqrt(jnp.maximum(force_magnitude, 1e-8))
        aggregated_forces = aggregated_forces + total_gradient_force / force_scale

    theoretical_target = anchored_preds_norm + aggregated_forces
    trainable_preds_norm = preds / jax.lax.stop_gradient(dim_scale)
    spatial_diff = trainable_preds_norm - jax.lax.stop_gradient(theoretical_target)
    dbp_loss = jnp.mean(jnp.square(spatial_diff), axis=(-1, -2))

    diagnostics = {k: jnp.mean(v) for k, v in diagnostics.items()}
    return dbp_loss, diagnostics
