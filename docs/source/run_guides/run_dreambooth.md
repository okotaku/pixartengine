# Stable Diffusion DremBooth Training

You can also check [`configs/dreambooth/README.md`](https://github.com/okotaku/pixartengine/tree/main/diffengine/configs/dreambooth/README.md) file.

## Configs

All configuration files are placed under the [`configs/dreambooth`](https://github.com/okotaku/pixartengine/tree/main/diffengine/configs/dreambooth/) folder.

Following is the example config from the pixart_alpha_512_dreambooth_lora_dog config file in [`configs/dreambooth/pixart_alpha_512_dreambooth_lora_dog.py`](https://github.com/okotaku/pixartengine/tree/main/diffengine/configs/dreambooth/pixart_alpha_512_dreambooth_lora_dog.py):

```
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.dog_dreambooth import *
    from .._base_.default_runtime import *
    from .._base_.models.pixart_alpha_512_lora import *
    from .._base_.schedules.diffusion_1k import *

model.update(weight_dtype="bf16")
```

## Run DreamBooth training

Run DreamBooth train

```
# single gpu
$ diffengine train ${CONFIG_FILE}
# Example
$ diffengine train pixart_alpha_512_dreambooth_lora_dog

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

checkpoint = Path('work_dirs/pixart_alpha_512_dreambooth_lora_dog/step999')
prompt = 'A photo of sks dog in a bucket'

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
