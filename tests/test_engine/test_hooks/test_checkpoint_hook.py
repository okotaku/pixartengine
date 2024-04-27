import copy
import shutil
from pathlib import Path

from diffusers import Transformer2DModel
from mmengine.model import BaseModel
from mmengine.registry import MODELS
from mmengine.testing import RunnerTestCase
from mmengine.testing.runner_test_case import ToyModel
from torch import nn

from diffengine.engine.hooks import CheckpointHook


class DummyWrapper(BaseModel):

    def __init__(self, model) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            model = MODELS.build(model)
        self.module = model

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class ToyModel2(ToyModel):

    def __init__(self) -> None:
        super().__init__()
        self.transformer = Transformer2DModel(
            sample_size=8,
            num_layers=2,
            patch_size=2,
            attention_head_dim=8,
            num_attention_heads=3,
            caption_channels=32,
            in_channels=4,
            cross_attention_dim=24,
            out_channels=8,
            attention_bias=True,
            activation_fn="gelu-approximate",
            num_embeds_ada_norm=1000,
            norm_type="ada_norm_single",
            norm_elementwise_affine=False,
            norm_eps=1e-6)
        self.vae = nn.Linear(2, 1)
        self.text_encoder = nn.Linear(2, 1)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class TestCheckpointHook(RunnerTestCase):

    def setUp(self) -> None:
        MODELS.register_module(name="DummyWrapper", module=DummyWrapper)
        MODELS.register_module(name="ToyModel2", module=ToyModel2)
        return super().setUp()

    def tearDown(self):
        MODELS.module_dict.pop("DummyWrapper")
        MODELS.module_dict.pop("ToyModel2")
        return super().tearDown()

    def test_init(self):
        CheckpointHook()

    def test_before_save_checkpoint(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model.type = "ToyModel2"
        runner = self.build_runner(cfg)
        checkpoint = dict(state_dict=ToyModel2().state_dict())
        hook = CheckpointHook()
        hook.before_save_checkpoint(runner, checkpoint)

        for key in checkpoint["state_dict"]:
            assert key.startswith(("transformer", "text_encoder"))

    def test_after_run(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model.type = "ToyModel2"
        runner = self.build_runner(cfg)
        hook = CheckpointHook()
        hook.after_run(runner)

        assert (Path(runner.work_dir) / (f"step{runner.iter}/transformer/"
                     "diffusion_pytorch_model.safetensors")).exists()
        shutil.rmtree(
            Path(runner.work_dir) / f"step{runner.iter}")
