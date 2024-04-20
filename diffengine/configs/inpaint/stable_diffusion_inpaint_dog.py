from mmengine.config import read_base

with read_base():
    from .._base_.datasets.dog_inpaint import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_inpaint import *
    from .._base_.schedules.stable_diffusion_1k import *

model.update(weight_dtype="bf16")
