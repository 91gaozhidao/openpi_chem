# `DBP-VA` 服务器测试指令与回传要求

## Summary

- 按下面顺序在服务器执行。
- **任意阶段一旦报错就先停下**，不要继续下一阶段。
- 每个阶段结束后，把指定日志文件的完整内容回传给我，我会据此决定下一步是继续验证、调配置，还是改代码。

## 0. 准备

先设置工作目录并建立统一日志目录：

```bash
export OPENPI_ROOT=/你的/openpi/绝对路径
cd "$OPENPI_ROOT"

mkdir -p /tmp/dbp_va_logs
```

下面所有命令都默认在这个目录下执行。

## 1. 环境确认

```bash
python -V 2>&1 | tee /tmp/dbp_va_logs/01_python.txt
python -c "import jax, flax, optax, tyro; print('jax', jax.__version__); print(jax.devices())" 2>&1 | tee /tmp/dbp_va_logs/02_jax.txt
python -c "from openpi.training import config as c; print(c.get_config('debug_va').name); print(c.get_config('dbp_va_chem').name); print(c.get_config('dbp_va_chem_smoke').name)" 2>&1 | tee /tmp/dbp_va_logs/03_configs.txt
python -c "from openpi.models import va_config, va, dbp_loss; print('imports ok')" 2>&1 | tee /tmp/dbp_va_logs/04_imports.txt
```

### 环境确认后回传

把这 4 个文件的完整内容发给我：

- `/tmp/dbp_va_logs/01_python.txt`
- `/tmp/dbp_va_logs/02_jax.txt`
- `/tmp/dbp_va_logs/03_configs.txt`
- `/tmp/dbp_va_logs/04_imports.txt`

## 2. 最小单元测试

优先跑完整版本：

```bash
pytest -q openpi/src/openpi/models/model_test.py -k va 2>&1 | tee /tmp/dbp_va_logs/05_model_test.txt
pytest -q openpi/src/openpi/training/weight_loaders_test.py 2>&1 | tee /tmp/dbp_va_logs/06_weight_loader_test.txt
```

如果机器资源紧张，改跑更窄的目标：

```bash
pytest -q openpi/src/openpi/models/model_test.py::test_va_model 2>&1 | tee /tmp/dbp_va_logs/05_model_test.txt
pytest -q openpi/src/openpi/training/weight_loaders_test.py::test_vision_backbone_checkpoint_weight_loader 2>&1 | tee /tmp/dbp_va_logs/06_weight_loader_test.txt
```

### 单元测试后回传

把这 2 个文件的完整内容发给我：

- `/tmp/dbp_va_logs/05_model_test.txt`
- `/tmp/dbp_va_logs/06_weight_loader_test.txt`

## 3. 数据链路与 norm stats

先检查 dataloader 输出：

```bash
python scripts/inspect_policy_batch.py --config-name debug_va 2>&1 | tee /tmp/dbp_va_logs/07_debug_batch.txt
python scripts/inspect_policy_batch.py --config-name dbp_va_chem --num-batches 1 2>&1 | tee /tmp/dbp_va_logs/08_chem_batch.txt
```

再做一次较小规模的 norm stats：

```bash
python scripts/compute_norm_stats.py --config-name dbp_va_chem --max_frames 512 2>&1 | tee /tmp/dbp_va_logs/09_norm_stats.txt
```

如果这一步通过，再确认 norm stats 写出目录：

```bash
find . -path "*dbp_va_chem*/*xixihaha*" | tee /tmp/dbp_va_logs/10_norm_stats_path.txt
```

### 数据链路后回传

把这 4 个文件的完整内容发给我：

- `/tmp/dbp_va_logs/07_debug_batch.txt`
- `/tmp/dbp_va_logs/08_chem_batch.txt`
- `/tmp/dbp_va_logs/09_norm_stats.txt`
- `/tmp/dbp_va_logs/10_norm_stats_path.txt`

## 4. 训练 smoke test

先跑最小 fake smoke：

```bash
python scripts/train.py debug_va --exp-name=smoke_debug_va --overwrite 2>&1 | tee /tmp/dbp_va_logs/11_train_debug_va.txt
```

再跑真实数据的 smoke config：

```bash
python scripts/train.py dbp_va_chem_smoke --exp-name=smoke_dbp_va_chem --overwrite 2>&1 | tee /tmp/dbp_va_logs/12_train_dbp_va_chem_smoke.txt
```

训练完成后确认 checkpoint 路径：

```bash
find . -path "*checkpoint*dbp_va_chem_smoke*smoke_dbp_va_chem*" | sort | tee /tmp/dbp_va_logs/13_checkpoint_paths.txt
```

### 训练 smoke 后回传

把这 3 个文件的完整内容发给我：

- `/tmp/dbp_va_logs/11_train_debug_va.txt`
- `/tmp/dbp_va_logs/12_train_dbp_va_chem_smoke.txt`
- `/tmp/dbp_va_logs/13_checkpoint_paths.txt`

## 5. checkpoint 恢复测试

```bash
python scripts/train.py dbp_va_chem_smoke --exp-name=smoke_dbp_va_chem --resume 2>&1 | tee /tmp/dbp_va_logs/14_resume.txt
```

## 6. 本地推理 smoke test

先把下面变量替换成上一步找到的真实 checkpoint step 目录：

```bash
export DBP_VA_SMOKE_CKPT=/绝对路径/到/checkpoint/dbp_va_chem_smoke/smoke_dbp_va_chem/<STEP>
echo "$DBP_VA_SMOKE_CKPT"
```

直接推理：

```bash
python scripts/smoke_policy_infer.py \
  --config-name dbp_va_chem_smoke \
  --checkpoint-dir "$DBP_VA_SMOKE_CKPT" \
  2>&1 | tee /tmp/dbp_va_logs/15_smoke_infer.txt
```

`ActionChunkBroker` 路径：

```bash
python scripts/smoke_policy_infer.py \
  --config-name dbp_va_chem_smoke \
  --checkpoint-dir "$DBP_VA_SMOKE_CKPT" \
  --broker \
  2>&1 | tee /tmp/dbp_va_logs/16_smoke_broker.txt
```

## 7. policy server 启动测试

单独开一个终端运行：

```bash
python scripts/serve_policy.py policy:checkpoint \
  --policy.config=dbp_va_chem_smoke \
  --policy.dir="$DBP_VA_SMOKE_CKPT" \
  2>&1 | tee /tmp/dbp_va_logs/17_serve_policy.txt
```

如果 server 能正常启动，就保持它运行；如果报错，停在这里并回传日志。

### 恢复与推理后回传

把这 4 个文件的完整内容发给我：

- `/tmp/dbp_va_logs/14_resume.txt`
- `/tmp/dbp_va_logs/15_smoke_infer.txt`
- `/tmp/dbp_va_logs/16_smoke_broker.txt`
- `/tmp/dbp_va_logs/17_serve_policy.txt`

## 8. 推理时延 benchmark

先准备 3 个 checkpoint：

```bash
export PI0_CKPT=/你的/pi0_chem/checkpoint_step目录
export PI05_CKPT=/你的/pi05_chem/checkpoint_step目录
export DBP_VA_CKPT=/你的/dbp_va_chem或dbp_va_chem_smoke/checkpoint_step目录
```

然后跑 benchmark：

```bash
python scripts/bench_policy_infer.py \
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

### benchmark 后回传

把这个文件完整发给我：

- `/tmp/dbp_va_logs/18_bench.txt`

## 判定规则

- 任意一步如果出现报错、`NaN`、shape mismatch、`ModuleNotFoundError`、checkpoint load failure，就**不要继续下一步**。
- 如果某一步通过，也按上面的要求回传对应日志，我会基于结果决定是否：
  - 调整 `config`
  - 修 `weight loader`
  - 修 `VA` 模型的 shape / mask / decoder
  - 修 `policy infer` 输入输出映射
  - 调整 benchmark 口径

## Assumptions

- 你在服务器上运行的是正确的 JAX 环境。
- `dbp_va_chem` 和 `dbp_va_chem_smoke` 的数据路径、视觉预训练路径在服务器上可访问。
- 当前先以 `dbp_va_chem_smoke` 完成完整链路验证，再决定是否切到完整 `dbp_va_chem`。
