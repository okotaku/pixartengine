from diffusers import AutoencoderKL, DDPMScheduler
from transformers import T5EncoderModel, T5Tokenizer

from diffengine.models.editors import PixArtDPO
from diffengine.models.transformers import Transformer2DModel
from diffengine.models.utils import OffsetNoise

base_model = "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
model = dict(
    type=PixArtDPO,
    model=base_model,
    tokenizer_max_length=300,
    enable_vb_loss=False,
    tokenizer=dict(type=T5Tokenizer.from_pretrained,
                subfolder="tokenizer"),
    scheduler=dict(type=DDPMScheduler.from_pretrained,
                subfolder="scheduler"),
    text_encoder=dict(type=T5EncoderModel.from_pretrained,
                    subfolder="text_encoder"),
    vae=dict(
        type=AutoencoderKL.from_pretrained,
        subfolder="vae"),
    transformer=dict(
        type=Transformer2DModel.from_pretrained,
        pretrained_model_name_or_path="PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
        subfolder="transformer",
        use_additional_conditions=False),
    noise_generator=dict(type=OffsetNoise, offset_weight=0.1))
