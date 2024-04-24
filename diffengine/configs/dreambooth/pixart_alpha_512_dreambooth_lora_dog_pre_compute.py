from mmengine.config import read_base

with read_base():
    from .._base_.datasets.dog_dreambooth_pre_compute import *
    from .._base_.default_runtime import *
    from .._base_.models.pixart_alpha_512_lora import *
    from .._base_.schedules.diffusion_1k import *

model.update(pre_compute_text_embeddings=True,
             weight_dtype="bf16")
