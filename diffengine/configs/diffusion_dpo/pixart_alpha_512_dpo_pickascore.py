from mmengine.config import read_base

with read_base():
    from .._base_.datasets.pickapicv2 import *
    from .._base_.default_runtime import *
    from .._base_.models.pixart_alpha_512_dpo import *
    from .._base_.schedules.diffusion_10k_dpo_baseline import *
