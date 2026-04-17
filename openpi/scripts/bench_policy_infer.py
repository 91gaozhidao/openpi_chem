import dataclasses
import json
import statistics
import time

import numpy as np
import tyro

from openpi.policies import policy_config
from openpi.training import config as _config


@dataclasses.dataclass
class RunSpec:
    config_name: str
    checkpoint_dir: str
    label: str | None = None


@dataclasses.dataclass
class Args:
    runs: list[RunSpec]
    state_dim: int = 7
    image_height: int = 224
    image_width: int = 224
    warmup: int = 20
    iters: int = 100
    seed: int = 0


def make_example(args: Args) -> dict:
    rng = np.random.default_rng(args.seed)
    return {
        "camera_head": rng.integers(0, 256, size=(args.image_height, args.image_width, 3), dtype=np.uint8),
        "camera_right_wrist": rng.integers(0, 256, size=(args.image_height, args.image_width, 3), dtype=np.uint8),
        "state": rng.standard_normal(args.state_dim).astype(np.float32),
    }


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def benchmark_run(spec: RunSpec, args: Args, example: dict) -> dict:
    config = _config.get_config(spec.config_name)
    policy = policy_config.create_trained_policy(config, spec.checkpoint_dir)

    for _ in range(args.warmup):
        policy.infer(example)

    timings = []
    model_timings = []
    for _ in range(args.iters):
        start = time.perf_counter()
        outputs = policy.infer(example)
        timings.append((time.perf_counter() - start) * 1000.0)
        model_timings.append(float(outputs.get("policy_timing", {}).get("infer_ms", float("nan"))))

    return {
        "label": spec.label or spec.config_name,
        "config_name": spec.config_name,
        "checkpoint_dir": spec.checkpoint_dir,
        "iters": args.iters,
        "warmup": args.warmup,
        "client_infer_ms_mean": statistics.fmean(timings),
        "client_infer_ms_p50": percentile(timings, 50),
        "client_infer_ms_p95": percentile(timings, 95),
        "model_infer_ms_mean": statistics.fmean(model_timings),
        "model_infer_ms_p50": percentile(model_timings, 50),
        "model_infer_ms_p95": percentile(model_timings, 95),
    }


def main(args: Args) -> None:
    example = make_example(args)
    results = [benchmark_run(run, args, example) for run in args.runs]
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Args))
