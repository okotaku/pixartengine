from mmengine.config import read_base

from diffengine.engine.hooks import PeftSaveHook

with read_base():
    from .._base_.datasets.pokemon import *
    from .._base_.default_runtime import *
    from .._base_.models.pixart_alpha_512_lavi_bridge import *
    from .._base_.schedules.diffusion_50e import *

model.update(enable_vb_loss=False)

default_hooks.update(dict(checkpoint=dict(save_optimizer=False)))

optim_wrapper.update(
    optimizer=dict(lr=1e-5),
    clip_grad=dict(max_norm=1.0))

custom_hooks = [
    dict(type=VisualizationHook,
         prompt=["yoda pokemon",
                 "A group of Exeggcute",
                 "A water dragon",
                 "A cheerful pink Chansey holding an egg."]),
    dict(type=PeftSaveHook),
]
