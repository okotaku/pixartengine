from unittest.mock import MagicMock

from mmengine.runner import EpochBasedTrainLoop
from mmengine.testing import RunnerTestCase

from diffengine.engine.hooks import VisualizationHook


class TestVisualizationHook(RunnerTestCase):

    def test_before_train(self):
        runner = MagicMock()

        # test epoch-based
        runner.train_loop = MagicMock(spec=EpochBasedTrainLoop)
        runner.epoch = 5
        hook = VisualizationHook(prompt=["a dog"],
            height=64,
            width=64)
        hook.before_train(runner)

    def test_before_train_with_condition(self):
        runner = MagicMock()

        # test epoch-based
        runner.train_loop = MagicMock(spec=EpochBasedTrainLoop)
        runner.epoch = 5
        hook = VisualizationHook(
            prompt=["a dog"], condition_image=["testdata/color.jpg"],
            height=64,
            width=64)
        hook.before_train(runner)

    def test_after_train_epoch(self):
        runner = MagicMock()

        # test epoch-based
        runner.train_loop = MagicMock(spec=EpochBasedTrainLoop)
        runner.epoch = 5
        hook = VisualizationHook(prompt=["a dog"],
            height=64,
            width=64)
        hook.after_train_epoch(runner)

    def test_after_train_epoch_with_condition(self):
        runner = MagicMock()

        # test epoch-based
        runner.train_loop = MagicMock(spec=EpochBasedTrainLoop)
        runner.epoch = 5
        hook = VisualizationHook(
            prompt=["a dog"], condition_image=["testdata/color.jpg"],
            height=64,
            width=64)
        hook.after_train_epoch(runner)

    def test_after_train_epoch_with_example_iamge(self):
        runner = MagicMock()

        # test epoch-based
        runner.train_loop = MagicMock(spec=EpochBasedTrainLoop)
        runner.epoch = 5
        hook = VisualizationHook(
            prompt=["a dog"], example_image=["testdata/color.jpg"],
            height=64,
            width=64)
        hook.after_train_epoch(runner)
