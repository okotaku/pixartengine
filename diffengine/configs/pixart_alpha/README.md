# PixArt-α

[PixArt-α](https://arxiv.org/abs/2310.00426)

## Abstract

The most advanced text-to-image (T2I) models require significant training costs (e.g., millions of GPU hours), seriously hindering the fundamental innovation for the AIGC community while increasing CO2 emissions. This paper introduces PIXART-α, a Transformer-based T2I diffusion model whose image generation quality is competitive with state-of-the-art image generators (e.g., Imagen, SDXL, and even Midjourney), reaching near-commercial application standards. Additionally, it supports high-resolution image synthesis up to 1024px resolution with low training cost, as shown in Figure 1 and 2. To achieve this goal, three core designs are proposed: (1) Training strategy decomposition: We devise three distinct training steps that separately optimize pixel dependency, text-image alignment, and image aesthetic quality; (2) Efficient T2I Transformer: We incorporate cross-attention modules into Diffusion Transformer (DiT) to inject text conditions and streamline the computation-intensive class-condition branch; (3) High-informative data: We emphasize the significance of concept density in text-image pairs and leverage a large Vision-Language model to auto-label dense pseudo-captions to assist text-image alignment learning. As a result, PIXART-α's training speed markedly surpasses existing large-scale T2I models, e.g., PIXART-α only takes 10.8% of Stable Diffusion v1.5's training time (675 vs. 6,250 A100 GPU days), saving nearly $300,000 ($26,000 vs. $320,000) and reducing 90% CO2 emissions. Moreover, compared with a larger SOTA model, RAPHAEL, our training cost is merely 1%. Extensive experiments demonstrate that PIXART-α excels in image quality, artistry, and semantic control. We hope PIXART-α will provide new insights to the AIGC community and startups to accelerate building their own high-quality yet low-cost generative models from scratch.

<div align=center>
<img src="https://github.com/okotaku/diffengine/assets/24734142/89eff9a2-9955-4f28-9917-f426c8a4edcd"/>
</div>

## Citation

```
@misc{chen2023pixartalpha,
      title={PixArt-$\alpha$: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis},
      author={Junsong Chen and Jincheng Yu and Chongjian Ge and Lewei Yao and Enze Xie and Yue Wu and Zhongdao Wang and James Kwok and Ping Luo and Huchuan Lu and Zhenguo Li},
      year={2023},
      eprint={2310.00426},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
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
$ diffengine train pixart_alpha_512_pokemon
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

```py
from pathlib import Path

import torch
from diffusers import PixArtAlphaPipeline, AutoencoderKL, Transformer2DModel

checkpoint = Path('work_dirs/pixart_alpha_512_pokemon')
prompt = 'A water dragon'

vae = AutoencoderKL.from_pretrained(
    'stabilityai/sd-vae-ft-ema',
    torch_dtype=torch.bfloat16,
)
transformer = Transformer2DModel.from_pretrained(
    checkpoint, subfolder='transformer', torch_dtype=torch.bfloat16)
pipe = PixArtAlphaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-512x512",
    vae=vae,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
).to("cuda")

img = pipe(
    prompt,
    width=512,
    height=512,
    num_inference_steps=20,
).images[0]
img.save("demo.png")
```

You can see more details on [`docs/source/run_guides/run.md`](../../docs/source/run_guides/run.md#inference-with-diffusers).

## Results Example

#### pixart_alpha_512_pokemon

![example1](<>)
