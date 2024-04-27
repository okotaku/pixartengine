# Stable Diffusion LoRA

[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

## Abstract

An important paradigm of natural language processing consists of large-scale pre-training on general domain data and adaptation to particular tasks or domains. As we pre-train larger models, full fine-tuning, which retrains all model parameters, becomes less feasible. Using GPT-3 175B as an example -- deploying independent instances of fine-tuned models, each with 175B parameters, is prohibitively expensive. We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times. LoRA performs on-par or better than fine-tuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3, despite having fewer trainable parameters, a higher training throughput, and, unlike adapters, no additional inference latency. We also provide an empirical investigation into rank-deficiency in language model adaptation, which sheds light on the efficacy of LoRA.

<div align=center>
<img src="https://github.com/okotaku/diffengine/assets/24734142/542189e1-e88f-4e80-9a51-de241b37d994"/>
</div>

## Citation

```
@inproceedings{
hu2022lora,
title={Lo{RA}: Low-Rank Adaptation of Large Language Models},
author={Edward J Hu and Yelong Shen and Phillip Wallis and Zeyuan Allen-Zhu and Yuanzhi Li and Shean Wang and Lu Wang and Weizhu Chen},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=nZeVKeeFYf9}
}
```

## Run Training

Run Training

```
# single gpu
$ diffengine train ${CONFIG_FILE}
# multi gpus
$ NPROC_PER_NODE=${GPU_NUM} diffengine train ${CONFIG_FILE}

# Example.
$ diffengine train stable_diffusion_v15_lora_pokemon_blip
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

```py
from pathlib import Path

import torch
from diffusers import PixArtAlphaPipeline, AutoencoderKL
from peft import PeftModel

checkpoint = Path('work_dirs/pixart_alpha_512_lora_pokemon/step20850')
prompt = 'A water dragon'

vae = AutoencoderKL.from_pretrained(
    'stabilityai/sd-vae-ft-ema',
    torch_dtype=torch.bfloat16,
)
pipe = PixArtAlphaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-512x512",
    vae=vae,
    torch_dtype=torch.bfloat16,
).to("cuda")
pipe.transformer = PeftModel.from_pretrained(pipe.transformer, checkpoint / "transformer", adapter_name="default")

img = pipe(
    prompt,
    width=512,
    height=512,
    num_inference_steps=20,
).images[0]
img.save("demo.png")
```

You can see more details on [LoRA docs](../../docs/source/run_guides/run_lora.md#inference-with-diffusers).

## Results Example

#### pixart_alpha_512_lora_pokemon

![example1](<>)
