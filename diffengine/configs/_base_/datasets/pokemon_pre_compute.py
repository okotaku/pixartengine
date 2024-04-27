import torchvision
from mmengine.dataset import DefaultSampler
from transformers import T5EncoderModel, T5Tokenizer

from diffengine.datasets import HFDatasetPreComputeEmbs
from diffengine.datasets.transforms import (
    PackInputs,
    RandomCrop,
    RandomHorizontalFlip,
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
    dict(type=PackInputs, input_keys=["img", "prompt_embeds", "attention_mask"]),
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type=HFDatasetPreComputeEmbs,
        dataset="diffusers/pokemon-gpt4-captions",
        text_hasher="text_pokemon_pixart_alpha_512",
        model="PixArt-alpha/PixArt-XL-2-512x512",
        tokenizer=dict(type=T5Tokenizer.from_pretrained,
                            subfolder="tokenizer"),
        text_encoder=dict(type=T5EncoderModel.from_pretrained,
                        subfolder="text_encoder"),
        proportion_empty_prompts=0.1,
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
