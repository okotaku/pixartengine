from mmengine.config import read_base

from diffengine.engine.hooks import (
    CheckpointHook,
    LCMEMAUpdateHook,
    VisualizationHook,
)

with read_base():
    from .._base_.datasets.pokemon_pre_compute import *
    from .._base_.default_runtime import *
    from .._base_.models.pixart_alpha_512_lcm import *
    from .._base_.schedules.diffusion_50e import *

model.update(pre_compute_text_embeddings=True,
             weight_dtype="bf16")

optim_wrapper.update(
    optimizer=dict(lr=1e-5))

custom_hooks = [
    dict(type=VisualizationHook,
         prompt=["yoda pokemon",
                 "A group of Exeggcute",
                 "A water dragon",
                 "A cheerful pink Chansey holding an egg."]),
    dict(type=CheckpointHook),
    dict(type=LCMEMAUpdateHook),
]
