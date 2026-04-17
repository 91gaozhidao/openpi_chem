import dataclasses
import json

import numpy as np
import tyro

from openpi.policies import policy_config
from openpi.training import config as _config
from openpi_client.action_chunk_broker import ActionChunkBroker


@dataclasses.dataclass
class Args:
    config_name: str
    checkpoint_dir: str
    state_dim: int = 7
    image_height: int = 224
    image_width: int = 224
    broker: bool = False
    broker_action_horizon: int = 16
    broker_steps: int | None = None
    seed: int = 0


def make_example(args: Args) -> dict:
    rng = np.random.default_rng(args.seed)
    return {
        "camera_head": rng.integers(0, 256, size=(args.image_height, args.image_width, 3), dtype=np.uint8),
        "camera_right_wrist": rng.integers(0, 256, size=(args.image_height, args.image_width, 3), dtype=np.uint8),
        "state": rng.standard_normal(args.state_dim).astype(np.float32),
    }


def main(args: Args) -> None:
    config = _config.get_config(args.config_name)
    policy = policy_config.create_trained_policy(config, args.checkpoint_dir)
    example = make_example(args)

    if not args.broker:
        outputs = policy.infer(example)
        payload = {
            "mode": "direct",
            "keys": sorted(outputs.keys()),
            "action_shape": tuple(np.asarray(outputs["action"]).shape),
            "policy_timing": outputs.get("policy_timing", {}),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    horizon = args.broker_steps or args.broker_action_horizon
    broker = ActionChunkBroker(policy, action_horizon=args.broker_action_horizon)
    step_shapes = []
    for _ in range(horizon):
        outputs = broker.infer(example)
        step_shapes.append(tuple(np.asarray(outputs["action"]).shape))

    payload = {
        "mode": "broker",
        "steps": horizon,
        "broker_action_horizon": args.broker_action_horizon,
        "model_action_horizon": config.model.action_horizon,
        "step_action_shapes": step_shapes,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Args))
