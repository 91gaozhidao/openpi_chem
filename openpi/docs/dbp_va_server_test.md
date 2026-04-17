# DBP-VA Server Test Plan

This document turns the `DBP-VA` server validation plan into concrete commands and helper scripts.

## 1. Environment Check

```bash
export OPENPI_ROOT=/absolute/path/to/openpi
cd "$OPENPI_ROOT"

python -V
python -c "import jax, flax, optax, tyro; print('jax', jax.__version__); print(jax.devices())"
python -c "from openpi.training import config as c; print(c.get_config('debug_va').name); print(c.get_config('dbp_va_chem').name); print(c.get_config('dbp_va_chem_smoke').name)"
python -c "from openpi.models import va_config, va, dbp_loss; print('imports ok')"
```

## 2. Unit Tests

```bash
pytest -q openpi/src/openpi/models/model_test.py -k va
pytest -q openpi/src/openpi/training/weight_loaders_test.py
```

If resources are limited:

```bash
pytest -q openpi/src/openpi/models/model_test.py::test_va_model
pytest -q openpi/src/openpi/training/weight_loaders_test.py::test_vision_backbone_checkpoint_weight_loader
```

## 3. Data Path and Norm Stats

Inspect a single batch:

```bash
python scripts/inspect_policy_batch.py --config-name debug_va
python scripts/inspect_policy_batch.py --config-name dbp_va_chem --num-batches 1
```

Compute normalization statistics:

```bash
python scripts/compute_norm_stats.py --config-name dbp_va_chem
```

Or a reduced pass:

```bash
python scripts/compute_norm_stats.py --config-name dbp_va_chem --max_frames 512
```

## 4. Training Smoke Tests

Fast structural smoke test:

```bash
python scripts/train.py debug_va --exp-name=smoke_debug_va --overwrite
```

Real-data smoke test with reduced training budget:

```bash
python scripts/train.py dbp_va_chem_smoke --exp-name=smoke_dbp_va_chem --overwrite
```

Full config smoke test if resources allow:

```bash
python scripts/train.py dbp_va_chem --exp-name=smoke_dbp_va_chem_full --overwrite
```

## 5. Checkpoint Resume and Inference

Resume:

```bash
python scripts/train.py dbp_va_chem_smoke --exp-name=smoke_dbp_va_chem --resume
```

Serve a checkpoint:

```bash
python scripts/serve_policy.py policy:checkpoint \
  --policy.config=dbp_va_chem_smoke \
  --policy.dir=/absolute/path/to/checkpoints/dbp_va_chem_smoke/smoke_dbp_va_chem/<STEP>
```

Local direct inference check:

```bash
python scripts/smoke_policy_infer.py \
  --config-name dbp_va_chem_smoke \
  --checkpoint-dir /absolute/path/to/checkpoints/dbp_va_chem_smoke/smoke_dbp_va_chem/<STEP>
```

ActionChunkBroker path:

```bash
python scripts/smoke_policy_infer.py \
  --config-name dbp_va_chem_smoke \
  --checkpoint-dir /absolute/path/to/checkpoints/dbp_va_chem_smoke/smoke_dbp_va_chem/<STEP> \
  --broker
```

## 6. Benchmark Comparison

Benchmark multiple checkpoints with the same random chemistry-style input:

```bash
python scripts/bench_policy_infer.py \
  --runs.0.config-name pi0_chem \
  --runs.0.checkpoint-dir /path/to/pi0_chem_ckpt \
  --runs.0.label pi0_chem \
  --runs.1.config-name pi05_chem \
  --runs.1.checkpoint-dir /path/to/pi05_chem_ckpt \
  --runs.1.label pi05_chem \
  --runs.2.config-name dbp_va_chem \
  --runs.2.checkpoint-dir /path/to/dbp_va_chem_ckpt \
  --runs.2.label dbp_va_chem \
  --warmup 20 \
  --iters 100
```

The script prints JSON with:

- `client_infer_ms_mean`
- `client_infer_ms_p50`
- `client_infer_ms_p95`
- `model_infer_ms_mean`
- `model_infer_ms_p50`
- `model_infer_ms_p95`

Use the same GPU, image size, camera count, and `state_dim` when comparing checkpoints.
