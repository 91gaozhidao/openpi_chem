import dataclasses
import json

import tyro

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


@dataclasses.dataclass
class Args:
    config_name: str
    num_batches: int = 1
    shuffle: bool = False


def main(args: Args) -> None:
    config = _config.get_config(args.config_name)
    loader = _data_loader.create_data_loader(
        config,
        shuffle=args.shuffle,
        num_batches=args.num_batches,
    )
    observation, actions = next(iter(loader))

    payload = {
        "config_name": args.config_name,
        "state_shape": tuple(observation.state.shape),
        "image_shapes": {key: tuple(value.shape) for key, value in observation.images.items()},
        "image_mask_shapes": {key: tuple(value.shape) for key, value in observation.image_masks.items()},
        "actions_shape": tuple(actions.shape),
        "has_tokenized_prompt": observation.tokenized_prompt is not None,
        "has_tokenized_prompt_mask": observation.tokenized_prompt_mask is not None,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Args))
