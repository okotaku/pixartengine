import torchvision
from mmengine.dataset import InfiniteSampler
from transformers import T5EncoderModel, T5Tokenizer

from diffengine.datasets import HFDreamBoothDatasetPreComputeEmbs
from diffengine.datasets.transforms import (
    PackInputs,
    RandomCrop,
    RandomHorizontalFlip,
    TorchVisonTransformWrapper,
)
from diffengine.engine.hooks import (
    CompileHook,
    PeftSaveHook,
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
    dict(type=PackInputs, input_keys=["img", "prompt_embeds", "attention_mask"]),
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type=HFDreamBoothDatasetPreComputeEmbs,
        dataset="diffusers/dog-example",
        instance_prompt="a photo of sks dog",
        model="PixArt-alpha/PixArt-XL-2-512x512",
        tokenizer=dict(type=T5Tokenizer.from_pretrained,
                            subfolder="tokenizer"),
        text_encoder=dict(type=T5EncoderModel.from_pretrained,
                        subfolder="text_encoder"),
        pipeline=train_pipeline),
    sampler=dict(type=InfiniteSampler, shuffle=True),
)

val_dataloader = None
val_evaluator = None
test_dataloader = val_dataloader
test_evaluator = val_evaluator

custom_hooks = [
    dict(
        type=VisualizationHook,
        prompt=["A photo of sks dog in a bucket"] * 4,
        by_epoch=False,
        interval=100),
    dict(type=PeftSaveHook),
    dict(type=CompileHook),
]
