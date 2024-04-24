from diffusers import AutoencoderKL, DDPMScheduler
from peft import LoraConfig
from transformers import AutoTokenizer, T5EncoderModel

from diffengine.models.editors import (
    LaViBridgeTextAdapter,
    PixArtLaViBridge,
)
from diffengine.models.transformers import Transformer2DModel

base_model = "PixArt-alpha/PixArt-XL-2-512x512"
llm_model = "t5-large"
model = dict(type=PixArtLaViBridge,
             model=base_model,
             tokenizer=dict(type=AutoTokenizer.from_pretrained,
                            pretrained_model_name_or_path=llm_model),
             scheduler=dict(type=DDPMScheduler.from_pretrained,
                            subfolder="scheduler"),
             text_encoder=dict(type=T5EncoderModel.from_pretrained,
                               pretrained_model_name_or_path=llm_model),
             adapter=dict(type=LaViBridgeTextAdapter,
                          in_dim=1024, int_dim=2560, out_dim=4096),
             vae=dict(
                type=AutoencoderKL.from_pretrained,
                pretrained_model_name_or_path="stabilityai/sd-vae-ft-ema"),
             transformer=dict(type=Transformer2DModel.from_pretrained,
                             subfolder="transformer"),
    text_encoder_lora_config=dict(
        type=LoraConfig,
        r=32,
        lora_alpha=32,
        target_modules=["q", "k", "v", "o"]),
    transformer_lora_config=dict(
        type=LoraConfig,
        r=32,
        lora_alpha=32,
        target_modules=["to_q", "to_v", "to_k", "to_out.0"]),
    gradient_checkpointing=True)
