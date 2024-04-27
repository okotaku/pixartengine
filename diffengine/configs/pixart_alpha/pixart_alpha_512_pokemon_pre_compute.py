from mmengine.config import read_base

with read_base():
    from .._base_.datasets.pokemon_pre_compute import *
    from .._base_.default_runtime import *
    from .._base_.models.pixart_alpha_512 import *
    from .._base_.schedules.diffusion_50e import *

model.update(pre_compute_text_embeddings=True,
             enable_vb_loss=False,
             weight_dtype="bf16")

optim_wrapper.update(
    optimizer=dict(lr=1e-5))
