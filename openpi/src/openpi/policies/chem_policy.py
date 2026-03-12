
import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(14),
        "observation/images/camera1_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/images/camera2_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/images/camera3_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the scoop in the cup",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class ChemInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct ele# chem
@dataclasses.dataclass(frozen=True)
class ChemDataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for chem).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/images/camera1_image": "observation/images/camera1_image",
                        "observation/images/camera2_image": "observation/images/camera2_image",
                        "observation/images/camera3_image": "observation/images/camera3_image",
                        "observation/state": "observation/state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `chem_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[chem_policy.ChemInputs(model_type=model_config.model_type)],
            outputs=[chem_policy.ChemOutputs()],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
# chem
ments of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        # import pdb;pdb.set_trace()
        # print(data.keys())
        
        camera1 = _parse_image(data["camera_head"])
        # camera2 = _parse_image(data["camera_left_wrist"])
        camera3 = _parse_image(data["camera_right_wrist"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["state"],
            "image": {
                "base_0_rgb": camera1,
                # "left_wrist_0_rgb": camera2,
                "left_wrist_0_rgb": np.zeros_like(camera3,dtype=np.uint8),
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": camera3,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                # "left_wrist_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_,   
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "action" in data:
            inputs["actions"] = data["action"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            # import ipdb; ipdb.set_trace()
            inputs["prompt"] = data["prompt"]
            # print("==========================================",data["prompt"])

        return inputs


@dataclasses.dataclass(frozen=True)
class ChemOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        # return {"action": np.asarray(data["actions"][:, :14])}
        return {"action": np.asarray(data["actions"][:, :7])}
