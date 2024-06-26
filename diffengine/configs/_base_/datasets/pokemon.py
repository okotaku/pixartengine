import torchvision
from mmengine.dataset import DefaultSampler

from diffengine.datasets import HFDataset
from diffengine.datasets.transforms import (
    PackInputs,
    RandomCrop,
    RandomHorizontalFlip,
    RandomTextDrop,
    T5TextPreprocess,
    TorchVisonTransformWrapper,
)
from diffengine.engine.hooks import (
    CheckpointHook,
    CompileHook,
    VisualizationHook,
)

train_pipeline = [
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.Resize,
         size=512, interpolation="bilinear"),
    dict(type=RandomCrop, size=512),
    dict(type=RandomHorizontalFlip, p=0.5),
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.ToTensor),
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.Normalize, mean=[0.5], std=[0.5]),
    dict(type=RandomTextDrop),
    dict(type=T5TextPreprocess),
    dict(type=PackInputs),
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type=HFDataset,
        dataset="diffusers/pokemon-gpt4-captions",
        pipeline=train_pipeline),
    sampler=dict(type=DefaultSampler, shuffle=True),
)

val_dataloader = None
val_evaluator = None
test_dataloader = val_dataloader
test_evaluator = val_evaluator

custom_hooks = [
    dict(type=VisualizationHook,
         prompt=["yoda pokemon",
                 "A group of Exeggcute",
                 "A water dragon",
                 "A cheerful pink Chansey holding an egg."]),
    dict(type=CheckpointHook),
    dict(type=CompileHook),
]
