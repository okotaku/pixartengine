from diffusers import AutoencoderKL, DDPMScheduler
from peft import LoraConfig
from transformers import T5EncoderModel, T5Tokenizer

from diffengine.models.editors import PixArt
from diffengine.models.transformers import Transformer2DModel

base_model = "PixArt-alpha/PixArt-XL-2-512x512"
model = dict(type=PixArt,
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
                             subfolder="transformer"),
    transformer_lora_config=dict(
        type=LoraConfig,
        r=32,
        lora_alpha=32,
        target_modules=["to_q", "to_v", "to_k", "to_out.0"]))
