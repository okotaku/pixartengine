from mmengine.config import read_base

with read_base():
    from .._base_.datasets.pokemon import *
    from .._base_.default_runtime import *
    from .._base_.models.pixart_alpha_512_ella import *
    from .._base_.schedules.diffusion_50e_baseline import *

model.update(enable_vb_loss=False)

default_hooks.update(dict(checkpoint=dict(save_optimizer=False)))

custom_hooks = [
    dict(type=VisualizationHook,
         prompt=["yoda pokemon",
                 "A group of Exeggcute",
                 "A water dragon",
                 "A cheerful pink Chansey holding an egg."]),
    dict(type=CheckpointHook),
]
