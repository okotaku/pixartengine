# PixArt LoRA Training

You can also check [`configs/lora/README.md`](https://github.com/okotaku/pixartengine/tree/main/diffengine/configs/lora/README.md) file.

## Configs

All configuration files are placed under the [`configs/lora`](https://github.com/okotaku/pixartengine/tree/main/diffengine/configs/lora/) folder.

Following is the example config from the pixart_alpha_512_lora_pokemon config file in [`configs/lora/pixart_alpha_512_lora_pokemon.py`](https://github.com/okotaku/pixartengine/tree/main/diffengine/configs/lora/pixart_alpha_512_lora_pokemon.py):

```
from mmengine.config import read_base

from diffengine.engine.hooks import PeftSaveHook

with read_base():
    from .._base_.datasets.pokemon import *
    from .._base_.default_runtime import *
    from .._base_.models.pixart_alpha_512_lora import *
    from .._base_.schedules.diffusion_50e import *

model.update(weight_dtype="bf16",
             enable_vb_loss=False)

custom_hooks = [
    dict(type=VisualizationHook,
         prompt=["yoda pokemon",
                 "A group of Exeggcute",
                 "A water dragon",
                 "A cheerful pink Chansey holding an egg."]),
    dict(type=PeftSaveHook),
    dict(type=CompileHook),
]
```

## Run LoRA training

Run LoRA training:

```
# single gpu
$ diffengine train ${CONFIG_FILE}
# Example
$ diffengine train pixart_alpha_512_lora_pokemon

# multi gpus
$ NPROC_PER_NODE=${GPU_NUM} diffengine train ${CONFIG_FILE}
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

```py
from pathlib import Path

import torch
from diffusers import PixArtAlphaPipeline, AutoencoderKL
from peft import PeftModel

checkpoint = Path('work_dirs/pixart_alpha_512_lora_pokemon/step20850')
prompt = 'A water dragon'

vae = AutoencoderKL.from_pretrained(
    'stabilityai/sd-vae-ft-ema',
    torch_dtype=torch.bfloat16,
)
pipe = PixArtAlphaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-512x512",
    vae=vae,
    torch_dtype=torch.bfloat16,
).to("cuda")
pipe.transformer = PeftModel.from_pretrained(pipe.transformer, checkpoint / "transformer", adapter_name="default")

img = pipe(
    prompt,
    width=512,
    height=512,
    num_inference_steps=20,
).images[0]
img.save("demo.png")
```
