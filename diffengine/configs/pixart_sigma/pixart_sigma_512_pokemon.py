from mmengine.config import read_base

with read_base():
    from .._base_.datasets.pokemon_baseline import *
    from .._base_.default_runtime import *
    from .._base_.models.pixart_sigma_512 import *
    from .._base_.schedules.diffusion_50e_baseline import *
