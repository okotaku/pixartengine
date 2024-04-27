from mmengine.config import read_base

from diffengine.engine.hooks import PeftSaveHook

with read_base():
    from .._base_.datasets.pokemon import *
    from .._base_.default_runtime import *
    from .._base_.models.pixart_alpha_512_lora import *
    from .._base_.schedules.diffusion_50e import *

model.update(weight_dtype="bf16",
             enable_vb_loss=False)

custom_hooks = [
    dict(type=VisualizationHook,
         prompt=["yoda pokemon",
                 "A group of Exeggcute",
                 "A water dragon",
                 "A cheerful pink Chansey holding an egg."]),
    dict(type=PeftSaveHook),
    dict(type=CompileHook),
]
