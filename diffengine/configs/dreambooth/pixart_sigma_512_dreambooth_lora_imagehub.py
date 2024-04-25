from mmengine.config import read_base

with read_base():
    from .._base_.datasets.imagehub_dreambooth import *
    from .._base_.default_runtime import *
    from .._base_.models.pixart_sigma_512_lora import *
    from .._base_.schedules.diffusion_1k_baseline import *

model.update(
    enable_vb_loss=False,
    #weight_dtype="bf16"
    )

optim_wrapper.update(
    optimizer=dict(lr=1e-4),
    #clip_grad=dict(max_norm=1.0),
    )
