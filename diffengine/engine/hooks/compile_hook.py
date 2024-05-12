import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner


class CompileHook(Hook):
    """Compile Hook.

    Args:
    ----
        backend (str): The backend to use for compilation.
            Defaults to "inductor".
        mode (str): The mode to use for compilation. Defaults to None.

    """

    priority = "VERY_LOW"

    def __init__(self, backend: str = "inductor", mode: str | None = None,
                 ) -> None:
        super().__init__()
        self.backend = backend
        self.mode = mode

    def before_train(self, runner: Runner) -> None:  # noqa: C901
        """Compile the model.

        Args:
        ----
            runner (Runner): The runner of the training process.

        """
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        if hasattr(model, "_forward_compile"):
            target = "_forward_compile"
            func = getattr(model, target)
            compiled_func = torch.compile(
                func, backend=self.backend, mode=self.mode)
            setattr(model, target, compiled_func)
        else:
            msg = "The model has no main network to compile."
            raise NotImplementedError(
                msg)

        if hasattr(model, "text_encoder"):
            model.text_encoder = torch.compile(
                model.text_encoder, backend=self.backend, mode=self.mode)
        if hasattr(model, "vae"):
            model.vae = torch.compile(
                model.vae, backend=self.backend, mode=self.mode)
        if hasattr(model, "image_encoder"):
            model.image_encoder = torch.compile(
                model.image_encoder, backend=self.backend, mode=self.mode)
        if hasattr(model, "orig_transformer"):
            model.orig_transformer = torch.compile(
                model.orig_transformer, backend=self.backend, mode=self.mode)
        if hasattr(model, "teacher_transformer"):
            model.teacher_transformer = torch.compile(
                model.teacher_transformer, backend=self.backend, mode=self.mode)
        if hasattr(model, "target_transformer"):
            model.target_transformer = torch.compile(
                model.target_transformer, backend=self.backend, mode=self.mode)
        if hasattr(model, "transformer_real"):
            model.transformer_real = torch.compile(
                model.transformer_real, backend=self.backend, mode=self.mode)
        if hasattr(model, "transformer_fake"):
            model.transformer_fake = torch.compile(
                model.transformer_fake, backend=self.backend, mode=self.mode)
