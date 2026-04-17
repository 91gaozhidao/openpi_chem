import flax.nnx as nnx
import jax
import numpy as np

from openpi.models import va_config
import openpi.models.model as _model
from openpi.training import weight_loaders


def test_vision_backbone_checkpoint_weight_loader(monkeypatch):
    config = va_config.VAConfig(
        vision_variant="mu/16",
        decoder_width=64,
        decoder_depth=2,
        decoder_num_heads=4,
        decoder_mlp_dim=128,
    )
    model = config.create(jax.random.key(0))
    params = nnx.state(model).to_pure_dict()

    img_subtree = params["PaliGemma"]["img"]

    def fake_restore_params(*args, **kwargs):
        del args, kwargs
        return {
            "PaliGemma": {
                "img": jax.tree.map(lambda x: np.ones(x.shape, dtype=x.dtype), img_subtree),
                "llm": {"unused": np.ones((1,), dtype=np.float32)},
            }
        }

    monkeypatch.setattr(_model, "restore_params", fake_restore_params)
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)

    loader = weight_loaders.VisionBackboneCheckpointWeightLoader("dummy")
    loaded = loader.load(params)

    assert "PaliGemma" in loaded
    assert "img" in loaded["PaliGemma"]
    assert "vision_proj" in loaded

    def assert_img_loaded(value):
        if isinstance(value, jax.ShapeDtypeStruct):
            raise AssertionError("Vision weights should be loaded as arrays.")
        np.testing.assert_array_equal(np.asarray(value), np.ones_like(np.asarray(value)))

    jax.tree.map(assert_img_loaded, loaded["PaliGemma"]["img"])
    np.testing.assert_array_equal(np.asarray(loaded["vision_proj"]["kernel"]), np.asarray(params["vision_proj"]["kernel"]))
    np.testing.assert_array_equal(np.asarray(loaded["vision_proj"]["bias"]), np.asarray(params["vision_proj"]["bias"]))
