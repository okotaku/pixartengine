from mmengine.hooks import CheckpointHook
from mmengine.optim import OptimWrapper
from optimi import AdamW

optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type=AdamW, lr=2e-6, weight_decay=3e-2, kahan_sum=True),
    clip_grad=dict(max_norm=0.01))

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=50)
val_cfg = None
test_cfg = None

default_hooks = dict(
    checkpoint=dict(
        type=CheckpointHook,
        interval=1,
        max_keep_ckpts=3,
        save_optimizer=True,
    ))
