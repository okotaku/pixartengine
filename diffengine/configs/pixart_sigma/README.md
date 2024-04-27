# PIXART-Σ

[PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation](https://arxiv.org/abs/2403.04692)

## Abstract

In this paper, we introduce PixArt-\\Sigma, a Diffusion Transformer model~(DiT) capable of directly generating images at 4K resolution. PixArt-\\Sigma represents a significant advancement over its predecessor, PixArt-\\alpha, offering images of markedly higher fidelity and improved alignment with text prompts. A key feature of PixArt-\\Sigma is its training efficiency. Leveraging the foundational pre-training of PixArt-\\alpha, it evolves from the `weaker' baseline to a `stronger' model via incorporating higher quality data, a process we term "weak-to-strong training". The advancements in PixArt-\\Sigma are twofold: (1) High-Quality Training Data: PixArt-\\Sigma incorporates superior-quality image data, paired with more precise and detailed image captions. (2) Efficient Token Compression: we propose a novel attention module within the DiT framework that compresses both keys and values, significantly improving efficiency and facilitating ultra-high-resolution image generation. Thanks to these improvements, PixArt-\\Sigma achieves superior image quality and user prompt adherence capabilities with significantly smaller model size (0.6B parameters) than existing text-to-image diffusion models, such as SDXL (2.6B parameters) and SD Cascade (5.1B parameters). Moreover, PixArt-\\Sigma's capability to generate 4K images supports the creation of high-resolution posters and wallpapers, efficiently bolstering the production of high-quality visual content in industries such as film and gaming.

<div align=center>
<img src="https://github.com/okotaku/diffengine/assets/24734142/48dd2a25-dc62-424b-b513-b2942a02a495"/>
</div>

## Citation

```
@misc{chen2024pixartsigma,
  title={PixArt-\Sigma: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation},
  author={Junsong Chen and Chongjian Ge and Enze Xie and Yue Wu and Lewei Yao and Xiaozhe Ren and Zhongdao Wang and Ping Luo and Huchuan Lu and Zhenguo Li},
  year={2024},
  eprint={2403.04692},
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
$ diffengine train pixart_sigma_512_pokemon
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

```py
```
