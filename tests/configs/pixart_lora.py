from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    Transformer2DModel,
)
from peft import LoraConfig
from transformers import AutoTokenizer, T5EncoderModel

from diffengine.models.editors import PixArt

base_model = "PixArt-alpha/PixArt-XL-2-1024-MS"
model = dict(
            type=PixArt,
             model=base_model,
             tokenizer=dict(
                 type=AutoTokenizer.from_pretrained,
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-t5"),
             scheduler=dict(type=DDPMScheduler.from_pretrained,
                            subfolder="scheduler"),
             text_encoder=dict(type=T5EncoderModel.from_pretrained,
                               pretrained_model_name_or_path="hf-internal-testing/tiny-random-t5"),
             vae=dict(
                type=AutoencoderKL),
             transformer=dict(
                type=Transformer2DModel,
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
                norm_eps=1e-6),
            unet_lora_config = dict(
                    type=LoraConfig, r=4,
                    target_modules=["to_q", "to_v", "to_k", "to_out.0"]),
            text_encoder_lora_config = dict(
                    type=LoraConfig, r=4,
                    target_modules=["q", "k", "v", "o"]))
