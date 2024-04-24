import copy

from diffusers import AutoencoderKL
from mmengine.config import Config
from mmengine.testing import RunnerTestCase
from transformers import T5EncoderModel

from diffengine.engine.hooks import CompileHook


class TestCompileHook(RunnerTestCase):

    def test_init(self) -> None:
        CompileHook()

    def test_before_train(self) -> None:
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model = Config.fromfile("tests/configs/pixart.py").model
        runner = self.build_runner(cfg)
        hook = CompileHook()
        func = runner.model._forward_compile
        assert runner.model._forward_compile == func
        assert isinstance(runner.model.vae, AutoencoderKL)
        assert isinstance(runner.model.text_encoder, T5EncoderModel)
        # compile
        hook.before_train(runner)
        assert runner.model._forward_compile != func
        assert not isinstance(runner.model.vae, AutoencoderKL)
        assert not isinstance(runner.model.text_encoder, T5EncoderModel)
