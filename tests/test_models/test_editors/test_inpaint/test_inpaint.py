import os
from unittest import TestCase

import pytest
import torch
from diffusers import AutoencoderKL, DDPMScheduler, Transformer2DModel
from mmengine.optim import OptimWrapper
from mmengine.registry import MODELS
from peft import LoraConfig
from torch.optim import SGD
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

from diffengine.models.editors import PixArtInpaint


class TestDiffusionInpaint(TestCase):

    def _get_config(self) -> dict:
        base_model = "PixArt-alpha/PixArt-XL-2-1024-MS"
        return dict(
            type=PixArtInpaint,
             model=base_model,
             tokenizer=dict(
                 type=AutoTokenizer.from_pretrained,
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-t5"),
             scheduler=dict(type=DDPMScheduler.from_pretrained,
                            pretrained_model_name_or_path=base_model,
                            subfolder="scheduler"),
             text_encoder=dict(type=T5EncoderModel.from_pretrained,
                               pretrained_model_name_or_path="hf-internal-testing/tiny-random-t5"),
             vae=dict(
                type=AutoencoderKL),
             transformer=dict(type=Transformer2DModel,
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
                norm_eps=1e-6))

    def _get_infer_config(self) -> dict:
        base_model = "PixArt-alpha/PixArt-XL-2-512x512"
        return dict(type=PixArtInpaint,
             model=base_model,
             tokenizer=dict(type=T5Tokenizer.from_pretrained,
                            subfolder="tokenizer"),
             scheduler=dict(type=DDPMScheduler.from_pretrained,
                            subfolder="scheduler"),
             text_encoder=dict(type=T5EncoderModel.from_pretrained,
                               subfolder="text_encoder"),
             vae=dict(
                type=AutoencoderKL.from_pretrained,
                pretrained_model_name_or_path="stabilityai/sd-vae-ft-ema"),
             transformer=dict(type=Transformer2DModel.from_pretrained,
                             subfolder="transformer"))

    def test_init(self):
        cfg = self._get_config()
        cfg.update(text_encoder_lora_config=dict(type="dummy"))
        with pytest.raises(
                AssertionError, match="If you want to use LoRA"):
            _ = MODELS.build(cfg)

        cfg = self._get_config()
        cfg.update(transformer_lora_config=dict(type="dummy"),
                finetune_text_encoder=True)
        with pytest.raises(
                AssertionError, match="If you want to finetune text"):
            _ = MODELS.build(cfg)

    @pytest.mark.skipif("GITHUB_ACTION" in os.environ,
                        reason="skip external api call during CI")
    def test_infer(self):
        cfg = self._get_infer_config()
        Diffuser = MODELS.build(cfg)
        Diffuser.to("cuda")

        # test infer
        result = Diffuser.infer(
            ["an insect robot preparing a delicious meal"],
            ["tests/testdata/color.jpg"],
            ["tests/testdata/color.jpg"],
            height=256,
            width=256,
            num_inference_steps=5)
        assert len(result) == 1
        assert result[0].shape == (256, 256, 3)

        # test device
        assert Diffuser.device.type == "cuda"

        # test infer with negative_prompt
        result = Diffuser.infer(
            ["an insect robot preparing a delicious meal"],
            ["tests/testdata/color.jpg"],
            ["tests/testdata/color.jpg"],
            negative_prompt="noise",
            height=256,
            width=256,
            num_inference_steps=5)
        assert len(result) == 1
        assert result[0].shape == (256, 256, 3)

        # test infer with latent output
        result = Diffuser.infer(
            ["an insect robot preparing a delicious meal"],
            ["tests/testdata/color.jpg"],
            ["tests/testdata/color.jpg"],
            output_type="latent",
            height=256,
            width=256,
            num_inference_steps=5)
        assert len(result) == 1
        assert type(result[0]) == torch.Tensor
        assert result[0].shape == (1, 4, 32, 32)

    @pytest.mark.skipif("GITHUB_ACTION" in os.environ,
                        reason="skip external api call during CI")
    def test_infer_with_lora(self):
        cfg = self._get_infer_config()
        cfg.update(
            transformer_lora_config=dict(
                type=LoraConfig, r=4,
                target_modules=["to_q", "to_v", "to_k", "to_out.0"]),
            text_encoder_lora_config = dict(
                type=LoraConfig, r=4,
                target_modules=["q", "k", "v", "o"]),
            finetune_text_encoder=True,
        )
        Diffuser = MODELS.build(cfg)
        Diffuser.to("cuda")

        # test infer
        result = Diffuser.infer(
            ["an insect robot preparing a delicious meal"],
            ["tests/testdata/color.jpg"],
            ["tests/testdata/color.jpg"],
            height=256,
            width=256,
            num_inference_steps=5)
        assert len(result) == 1
        assert result[0].shape == (256, 256, 3)

    @pytest.mark.skipif("GITHUB_ACTION" in os.environ,
                        reason="skip external api call during CI")
    def test_infer_with_pre_compute_embs(self):
        cfg = self._get_infer_config()
        cfg.update(pre_compute_text_embeddings=True)
        Diffuser = MODELS.build(cfg)
        Diffuser.to("cuda")

        assert not hasattr(Diffuser, "tokenizer")
        assert not hasattr(Diffuser, "text_encoder")

        # test infer
        result = Diffuser.infer(
            ["an insect robot preparing a delicious meal"],
            ["tests/testdata/color.jpg"],
            ["tests/testdata/color.jpg"],
            height=256,
            width=256,
            num_inference_steps=5)
        assert len(result) == 1
        assert result[0].shape == (256, 256, 3)

    def test_train_step(self):
        # test load with loss module
        cfg = self._get_config()
        Diffuser = MODELS.build(cfg)

        # test train step
        data = dict(
            inputs=dict(img=[torch.zeros((3, 64, 64))],
                        mask=[torch.zeros((1, 64, 64))],
                        masked_image=[torch.zeros((3, 64, 64))],
                        text=["a dog"]))
        optimizer = SGD(Diffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = Diffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_with_lora(self):
        # test load with loss module
        cfg = self._get_config()
        cfg.update(
            transformer_lora_config=dict(
                type=LoraConfig, r=4,
                target_modules=["to_q", "to_v", "to_k", "to_out.0"]),
            text_encoder_lora_config=dict(
                type=LoraConfig, r=4,
                target_modules=["q", "k", "v", "o"]),
        )
        Diffuser = MODELS.build(cfg)

        # test train step
        data = dict(
            inputs=dict(img=[torch.zeros((3, 64, 64))],
                        mask=[torch.zeros((1, 64, 64))],
                        masked_image=[torch.zeros((3, 64, 64))],
                        text=["a dog"]))
        optimizer = SGD(Diffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = Diffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_with_pre_compute_embs(self):
        # test load with loss module
        cfg = self._get_config()
        cfg.update(pre_compute_text_embeddings=True)
        Diffuser = MODELS.build(cfg)

        assert not hasattr(Diffuser, "tokenizer")
        assert not hasattr(Diffuser, "text_encoder")

        # test train step
        data = dict(
            inputs=dict(
                img=[torch.zeros((3, 64, 64))],
                mask=[torch.zeros((1, 64, 64))],
                masked_image=[torch.zeros((3, 64, 64))],
                prompt_embeds=[torch.zeros((77, 32))],
                attention_mask=[torch.zeros((77,))]))
        optimizer = SGD(Diffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = Diffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_val_and_test_step(self):
        cfg = self._get_config()
        Diffuser = MODELS.build(cfg)

        # test val_step
        with pytest.raises(NotImplementedError, match="val_step is not"):
            Diffuser.val_step(torch.zeros((1, )))

        # test test_step
        with pytest.raises(NotImplementedError, match="test_step is not"):
            Diffuser.test_step(torch.zeros((1, )))
