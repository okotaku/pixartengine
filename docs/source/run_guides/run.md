# PixArt Training

You can also check [`configs/pixart_alpha/README.md`](https://github.com/okotaku/pixartengine/tree/main/diffengine/configs/pixart_alpha/README.md) file.

## Configs

All configuration files are placed under the [`configs/pixart_alpha`](https://github.com/okotaku/pixartengine/blob/main/diffengine/configs/pixart_alpha) folder.

Following is the example config from the pixart_alpha_512_pokemon config file in [`configs/pixart_alpha/pixart_alpha_512_pokemon.py`](https://github.com/okotaku/pixartengine/blob/main/diffengine/configs/pixart_alpha/pixart_alpha_512_pokemon.py):

```
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.pokemon_baseline import *
    from .._base_.default_runtime import *
    from .._base_.models.pixart_alpha_512 import *
    from .._base_.schedules.diffusion_50e_baseline import *
```

#### Finetuning the text encoder and UNet

The script also allows you to finetune the text_encoder along with the unet.

```
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.pokemon_baseline import *
    from .._base_.default_runtime import *
    from .._base_.models.pixart_alpha_512 import *
    from .._base_.schedules.diffusion_50e_baseline import *

model.update(finetune_text_encoder=True)  # fine tune text encoder
```

#### Finetuning with Unet EMA

The script also allows you to finetune with Unet EMA.

```
from mmengine.config import read_base
from diffengine.engine.hooks import EMAHook

with read_base():
    from .._base_.datasets.pokemon_baseline import *
    from .._base_.default_runtime import *
    from .._base_.models.pixart_alpha_512 import *
    from .._base_.schedules.diffusion_50e_baseline import *

custom_hooks = [  # Hook is list, we should write all custom_hooks again.
    dict(type=VisualizationHook, prompt=['yoda pokemon'] * 4),
    dict(type=CheckpointHook),
    dict(type=EMAHook, ema_key="transformer", momentum=1e-4, priority='ABOVE_NORMAL')  # setup EMA Hook
]
```

#### Finetuning with other losses

The script also allows you to finetune with [Min-SNR Weighting Strategy](https://arxiv.org/abs/2303.09556).

```
from mmengine.config import read_base
from diffengine.models.losses import SNRL2Loss

with read_base():
    from .._base_.datasets.pokemon_baseline import *
    from .._base_.default_runtime import *
    from .._base_.models.pixart_alpha_512 import *
    from .._base_.schedules.diffusion_50e_baseline import *

model.update(loss=dict(type=SNRL2Loss, snr_gamma=5.0, loss_weight=1.0))  # setup Min-SNR Weighting Strategy
```

#### Finetuning with other noises

The script also allows you to finetune with [OffsetNoise](https://www.crosslabs.org/blog/diffusion-with-offset-noise).

```
from mmengine.config import read_base
from diffengine.models.utils import OffsetNoise

with read_base():
    from .._base_.datasets.pokemon_baseline import *
    from .._base_.default_runtime import *
    from .._base_.models.pixart_alpha_512 import *
    from .._base_.schedules.diffusion_50e_baseline import *

model.update(noise_generator=dict(type=OffsetNoise, offset_weight=0.05))  # setup OffsetNoise
```

#### Finetuning with other timesteps

The script also allows you to finetune with EarlierTimeSteps.

```
from mmengine.config import read_base
from diffengine.models.utils import EarlierTimeSteps

with read_base():
    from .._base_.datasets.pokemon_baseline import *
    from .._base_.default_runtime import *
    from .._base_.models.pixart_alpha_512 import *
    from .._base_.schedules.diffusion_50e_baseline import *

model.update(timesteps_generator=dict(type=EarlierTimeSteps))  # setup EarlierTimeSteps
```

#### Finetuning with pre-computed text embeddings

The script also allows you to finetune with pre-computed text embeddings.

```
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.pokemon_pre_compute import *
    from .._base_.default_runtime import *
    from .._base_.models.pixart_alpha_512 import *
    from .._base_.schedules.diffusion_50e import *

model.update(pre_compute_text_embeddings=True)
```

## Run training

Run train

```
# single gpu
$ diffengine train ${CONFIG_FILE}
# Example
$ diffengine train pixart_alpha_512_pokemon

# multi gpus
$ NPROC_PER_NODE=${GPU_NUM} diffengine train ${CONFIG_FILE}
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

```py
from pathlib import Path

import torch
from diffusers import PixArtAlphaPipeline, AutoencoderKL, Transformer2DModel

checkpoint = Path('work_dirs/pixart_alpha_512_pokemon')
prompt = 'A water dragon'

vae = AutoencoderKL.from_pretrained(
    'stabilityai/sd-vae-ft-ema',
    torch_dtype=torch.bfloat16,
)
transformer = Transformer2DModel.from_pretrained(
    checkpoint, subfolder='transformer', torch_dtype=torch.bfloat16)
pipe = PixArtAlphaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-512x512",
    vae=vae,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
).to("cuda")

img = pipe(
    prompt,
    width=512,
    height=512,
    num_inference_steps=20,
).images[0]
img.save("demo.png")
```

## Convert weights for diffusers format

You can convert weights for diffusers format. The converted weights will be saved in the specified directory.

```bash
$ diffengine convert ${CONFIG_FILE} ${INPUT_FILENAME} ${OUTPUT_DIR} --save-keys ${SAVE_KEYS}
# Example
$ diffengine convert pixart_alpha_512_pokemon work_dirs/pixart_alpha_512_pokemon/epoch_50.pth work_dirs/pixart_alpha_512_pokemon --save-keys transformer
```
