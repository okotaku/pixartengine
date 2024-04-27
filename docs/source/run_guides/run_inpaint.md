# PixArt Inpaint Training

You can also check [`configs/inpaint/README.md`](https://github.com/okotaku/pixartengine/tree/main/diffengine/configs/inpaint/README.md) file.

## Configs

All configuration files are placed under the [`configs/inpaint`](https://github.com/okotaku/pixartengine/tree/main/diffengine/configs/inpaint/) folder.

Following is the example config from the pixart_alpha_512_inpaint_dog config file in [`configs/inpaint/pixart_alpha_512_inpaint_dog.py`](https://github.com/okotaku/pixartengine/tree/main/diffengine/configs/inpaint/pixart_alpha_512_inpaint_dog.py):

```
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.dog_inpaint import *
    from .._base_.default_runtime import *
    from .._base_.models.pixart_alpha_512_inpaint import *
    from .._base_.schedules.diffusion_1k import *

model.update(weight_dtype="bf16")
```

## Run training

Run train

```
# single gpu
$ diffengine train ${CONFIG_FILE}
# Example
$ diffengine train pixart_alpha_512_inpaint_dog

# multi gpus
$ NPROC_PER_NODE=${GPU_NUM} diffengine train ${CONFIG_FILE}
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

```py
```
