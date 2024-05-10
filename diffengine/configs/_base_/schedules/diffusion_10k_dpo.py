from mmengine.hooks import CheckpointHook
from mmengine.optim import OptimWrapper
from mmengine.optim.scheduler import CosineAnnealingLR, LinearLR
from mmengine.runner import IterBasedTrainLoop
from optimi import AdamW

optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type=AdamW, lr=1e-5, weight_decay=3e-2, kahan_sum=True),
    clip_grad=dict(max_norm=1.0),
    accumulative_counts=256)

total_steps = 2000
param_scheduler = [
    dict(type=LinearLR, start_factor=0.01, by_epoch=False, begin=0,
         end=128 * total_steps//100),
    # Use a cosine learning rate at [100, 900) iterations
    dict(
        type=CosineAnnealingLR,
        T_max=128 * total_steps - 128 * total_steps//100,
        by_epoch=False,
        begin=128 * 10,
        end=128 * total_steps),
]

# train, val, test setting
train_cfg = dict(type=IterBasedTrainLoop, max_iters=128 * total_steps)
val_cfg = None
test_cfg = None

default_hooks = dict(
    checkpoint=dict(
        type=CheckpointHook,
        interval=10000,
        by_epoch=False,
        max_keep_ckpts=3,
        save_optimizer=True,
    ))
log_processor = dict(by_epoch=False)
