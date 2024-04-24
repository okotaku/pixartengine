import os.path as osp
from collections import OrderedDict

import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner


class CheckpointHook(Hook):
    """Delete 'vae' from checkpoint for efficient save."""

    priority = "VERY_LOW"

    def before_save_checkpoint(self, runner: Runner, checkpoint: dict) -> None:
        """Before save checkpoint hook.

        Args:
        ----
            runner (Runner): The runner of the training, validation or testing
                process.
            checkpoint (dict): Model's checkpoint.

        """
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        new_ckpt = OrderedDict()
        sd_keys = checkpoint["state_dict"].keys()
        for k in sd_keys:
            if not checkpoint["state_dict"][k].requires_grad:
                continue
            new_k = k.replace("._orig_mod", "")
            if k.startswith(("transformer", "adapter")):
                new_ckpt[new_k] = checkpoint["state_dict"][k]
            elif k.startswith("text_encoder") and hasattr(
                    model,
                    "finetune_text_encoder",
            ) and model.finetune_text_encoder:
                # if not finetune text_encoder, then not save
                new_ckpt[new_k] = checkpoint["state_dict"][k]
        checkpoint["state_dict"] = new_ckpt

    def after_run(self, runner: Runner) -> None:
        """After run hook."""
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        ckpt_path = osp.join(runner.work_dir, f"step{runner.iter}")
        if hasattr(model, "transformer"):
            for p in model.transformer.parameters():
                is_contiguous = p.is_contiguous()
                break
            if not is_contiguous:
                model.transformer = model.transformer.to(
                    memory_format=torch.contiguous_format)
            model.transformer.save_pretrained(
                osp.join(ckpt_path, "transformer"))
        if hasattr(
                    model,
                    "finetune_text_encoder",
            ) and model.finetune_text_encoder:
            model.text_encoder.save_pretrained(
                osp.join(ckpt_path, "text_encoder"))
        if hasattr(model, "adapter"):
            model.adapter.save_pretrained(osp.join(ckpt_path, "adapter"))
