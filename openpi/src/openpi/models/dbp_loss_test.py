import jax.numpy as jnp
import numpy as np
import torch

from openpi.models import dbp_loss
from openpi.models_pytorch.DBP_loss import compute_dbp_loss as compute_dbp_loss_torch


def test_dbp_loss_jax_matches_pytorch():
    rng = np.random.default_rng(0)

    preds = rng.standard_normal((2, 4, 12), dtype=np.float32)
    pos_targets = rng.standard_normal((2, 1, 12), dtype=np.float32)
    neg_targets = rng.standard_normal((2, 3, 12), dtype=np.float32)
    w_pred = rng.random((2, 4), dtype=np.float32)
    w_pos = rng.random((2, 1), dtype=np.float32)
    w_neg = rng.random((2, 3), dtype=np.float32)
    temperatures = (0.02, 0.05, 0.2)

    loss_jax, diagnostics_jax = dbp_loss.compute_dbp_loss(
        preds=jnp.asarray(preds),
        pos_targets=jnp.asarray(pos_targets),
        neg_targets=jnp.asarray(neg_targets),
        w_pred=jnp.asarray(w_pred),
        w_pos=jnp.asarray(w_pos),
        w_neg=jnp.asarray(w_neg),
        temp_schedule=temperatures,
    )
    loss_torch, diagnostics_torch = compute_dbp_loss_torch(
        preds=torch.from_numpy(preds),
        pos_targets=torch.from_numpy(pos_targets),
        neg_targets=torch.from_numpy(neg_targets),
        w_pred=torch.from_numpy(w_pred),
        w_pos=torch.from_numpy(w_pos),
        w_neg=torch.from_numpy(w_neg),
        temp_schedule=temperatures,
    )

    np.testing.assert_allclose(np.asarray(loss_jax), loss_torch.detach().cpu().numpy(), rtol=1e-2, atol=1e-3)
    assert diagnostics_jax.keys() == diagnostics_torch.keys()
    for key in diagnostics_jax:
        np.testing.assert_allclose(
            np.asarray(diagnostics_jax[key]),
            diagnostics_torch[key].detach().cpu().numpy(),
            rtol=1e-2,
            atol=1e-3,
        )
