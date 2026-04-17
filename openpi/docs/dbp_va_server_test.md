# `DBP-VA` 服务器测试指令与回传要求

## Summary

- 当前已完成：
  - 环境确认
  - `VA` 模型单元测试
  - `vision-only weight loader` 测试
  - chemistry dataloader / `norm stats`
  - `debug_va` 与 `dbp_va_chem_smoke` 训练 smoke test
- 当前推荐输入 checkpoint：
  - `/data1/workspace/gaoyuxuan/openpi_chem/openpi/checkpoint/dbp_va_chem_smoke/smoke_dbp_va_chem/3`
- 当前推荐运行约束：
  - 单卡：`CUDA_VISIBLE_DEVICES=4`
  - 保留 FFmpeg 动态库路径：`export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"`
  - 训练和恢复阶段统一加：`--num-workers=0`
- 已知观察：
  - `dbp_va_chem_smoke` 训练 smoke 中 `loss=9.0000` 持平
  - 该问题暂不阻塞 inference / serving contract 验证

## 0. 准备

```bash
export OPENPI_ROOT=/你的/openpi/绝对路径
cd "$OPENPI_ROOT"

mkdir -p /tmp/dbp_va_logs
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

下面所有命令默认在这个目录执行。

## 1. 环境确认

```bash
python -V 2>&1 | tee /tmp/dbp_va_logs/01_python.txt
python -c "import jax, flax, optax, tyro; print('jax', jax.__version__); print(jax.devices())" 2>&1 | tee /tmp/dbp_va_logs/02_jax.txt
python -c "from openpi.training import config as c; print(c.get_config('debug_va').name); print(c.get_config('dbp_va_chem').name); print(c.get_config('dbp_va_chem_smoke').name)" 2>&1 | tee /tmp/dbp_va_logs/03_configs.txt
python -c "from openpi.models import va_config, va, dbp_loss; print('imports ok')" 2>&1 | tee /tmp/dbp_va_logs/04_imports.txt
```

## 2. 最小单元测试

```bash
pytest -q src/openpi/models/model_test.py -k va 2>&1 | tee /tmp/dbp_va_logs/05_model_test.txt
pytest -q src/openpi/training/weight_loaders_test.py 2>&1 | tee /tmp/dbp_va_logs/06_weight_loader_test.txt
```

## 3. 数据链路与 `norm stats`

### `debug_va` dataloader shape 检查

```bash
python -c "
import json, dataclasses
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
cfg = dataclasses.replace(_config.get_config('debug_va'), num_workers=0)
loader = _data_loader.create_data_loader(
    cfg,
    shuffle=False,
    num_batches=1,
    skip_norm_stats=True,
    framework='pytorch',
)
obs, act = next(iter(loader))
payload = {
    'config_name': 'debug_va',
    'state_shape': tuple(obs.state.shape),
    'image_shapes': {k: tuple(v.shape) for k, v in obs.images.items()},
    'image_mask_shapes': {k: tuple(v.shape) for k, v in obs.image_masks.items()},
    'actions_shape': tuple(act.shape),
    'has_tokenized_prompt': obs.tokenized_prompt is not None,
    'has_tokenized_prompt_mask': obs.tokenized_prompt_mask is not None,
}
print(json.dumps(payload, indent=2, sort_keys=True))
" 2>&1 | tee /tmp/dbp_va_logs/07_debug_batch.txt
```

### `dbp_va_chem` dataloader shape 检查

```bash
python -c "
import json, dataclasses
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
cfg = dataclasses.replace(_config.get_config('dbp_va_chem'), num_workers=0)
loader = _data_loader.create_data_loader(
    cfg,
    shuffle=False,
    num_batches=1,
    skip_norm_stats=True,
    framework='pytorch',
)
obs, act = next(iter(loader))
payload = {
    'config_name': 'dbp_va_chem',
    'state_shape': tuple(obs.state.shape),
    'image_shapes': {k: tuple(v.shape) for k, v in obs.images.items()},
    'image_mask_shapes': {k: tuple(v.shape) for k, v in obs.image_masks.items()},
    'actions_shape': tuple(act.shape),
    'has_tokenized_prompt': obs.tokenized_prompt is not None,
    'has_tokenized_prompt_mask': obs.tokenized_prompt_mask is not None,
}
print(json.dumps(payload, indent=2, sort_keys=True))
" 2>&1 | tee /tmp/dbp_va_logs/08_chem_batch.txt
```

### `norm stats`

```bash
python scripts/compute_norm_stats.py --config-name dbp_va_chem --max_frames 512 2>&1 | tee /tmp/dbp_va_logs/09_norm_stats.txt
```

## 4. 训练 smoke test

### `debug_va`

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/train.py debug_va \
  --exp-name=smoke_debug_va \
  --overwrite \
  --num-workers=0 \
  2>&1 | tee /tmp/dbp_va_logs/11_train_debug_va.txt
```

### `dbp_va_chem_smoke`

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/train.py dbp_va_chem_smoke \
  --exp-name=smoke_dbp_va_chem \
  --overwrite \
  --num-workers=0 \
  2>&1 | tee /tmp/dbp_va_logs/12_train_dbp_va_chem_smoke.txt
```

### checkpoint 路径

```bash
find /data1/workspace/gaoyuxuan/openpi_chem/openpi/checkpoint -path "*dbp_va_chem_smoke*smoke_dbp_va_chem*" | sort | tee /tmp/dbp_va_logs/13_checkpoint_paths.txt
```

## 5. 恢复与推理 smoke test

### checkpoint 变量

```bash
export DBP_VA_SMOKE_CKPT=/data1/workspace/gaoyuxuan/openpi_chem/openpi/checkpoint/dbp_va_chem_smoke/smoke_dbp_va_chem/3
echo "$DBP_VA_SMOKE_CKPT"
```

### 恢复测试

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/train.py dbp_va_chem_smoke \
  --exp-name=smoke_dbp_va_chem \
  --resume \
  --num-workers=0 \
  2>&1 | tee /tmp/dbp_va_logs/14_resume.txt
```

### 本地直接推理

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/smoke_policy_infer.py \
  --config-name dbp_va_chem_smoke \
  --checkpoint-dir "$DBP_VA_SMOKE_CKPT" \
  2>&1 | tee /tmp/dbp_va_logs/15_smoke_infer.txt
```

### `ActionChunkBroker`

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/smoke_policy_infer.py \
  --config-name dbp_va_chem_smoke \
  --checkpoint-dir "$DBP_VA_SMOKE_CKPT" \
  --broker \
  2>&1 | tee /tmp/dbp_va_logs/16_smoke_broker.txt
```

### `policy server`

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/serve_policy.py policy:checkpoint \
  --policy.config=dbp_va_chem_smoke \
  --policy.dir="$DBP_VA_SMOKE_CKPT" \
  2>&1 | tee /tmp/dbp_va_logs/17_serve_policy.txt
```

成功标准：

- `14_resume.txt` 能识别已有 checkpoint，并正常恢复
- `15_smoke_infer.txt` 返回 `action` 与 `policy_timing`
- `16_smoke_broker.txt` 连续返回 `action_horizon` 次 action shape
- `17_serve_policy.txt` 能正常起服务，不报 checkpoint / transform / policy config 错误

## 6. 推理时延 benchmark

```bash
export PI0_CKPT=/你的/pi0_chem/checkpoint_step目录
export PI05_CKPT=/你的/pi05_chem/checkpoint_step目录
export DBP_VA_CKPT=/你的/dbp_va_chem或dbp_va_chem_smoke/checkpoint_step目录
```

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/bench_policy_infer.py \
  --runs.0.config-name pi0_chem \
  --runs.0.checkpoint-dir "$PI0_CKPT" \
  --runs.0.label pi0_chem \
  --runs.1.config-name pi05_chem \
  --runs.1.checkpoint-dir "$PI05_CKPT" \
  --runs.1.label pi05_chem \
  --runs.2.config-name dbp_va_chem \
  --runs.2.checkpoint-dir "$DBP_VA_CKPT" \
  --runs.2.label dbp_va_chem \
  --warmup 20 \
  --iters 100 \
  2>&1 | tee /tmp/dbp_va_logs/18_bench.txt
```

## 回传要求

### 当前下一轮最小回传

先回传下面 4 个文件：

- `/tmp/dbp_va_logs/14_resume.txt`
- `/tmp/dbp_va_logs/15_smoke_infer.txt`
- `/tmp/dbp_va_logs/16_smoke_broker.txt`
- `/tmp/dbp_va_logs/17_serve_policy.txt`

### 如果进入 benchmark

- `/tmp/dbp_va_logs/18_bench.txt`

## 判定规则

- 任意一步如果出现报错、`NaN`、shape mismatch、checkpoint load failure、serve 启动失败，就停止并回传当前阶段日志
- `loss=9.0000` 当前只记录为算法风险，不阻塞 inference / serving contract 验证
- 如果 `infer / broker / serve` 全通过，下一阶段再单独做 `DBP loss` 诊断
