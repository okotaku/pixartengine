import argparse
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from diffusers import PixArtSigmaPipeline, Transformer2DModel
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the IP adapter on a set of images.")
    parser.add_argument("--model", help="Model name",
                        type=str, default="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS")
    return parser.parse_args()


def main():
    try:
        import hpsv2
    except ImportError as e:
        msg = "Please install hpsv2"
        raise ImportError(msg) from e
    args = parse_args()

    model_name = args.model.split("/")[-1]
    out_dir = f"work_dirs/hpdv2_{model_name}"
    Path(out_dir).mkdir(exist_ok=True, parents=True)

    transformer = Transformer2DModel.from_pretrained(
        args.model,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        use_additional_conditions=False,
    )
    pipe = PixArtSigmaPipeline.from_pretrained(
        "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
        transformer=transformer,
        torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    eval_ds = load_dataset("ymhao/HPDv2", split="test")
    generator = torch.Generator(device="cuda").manual_seed(0)
    results = []
    for i, d in tqdm(enumerate(eval_ds)):
        img = pipe(d["prompt"], generator=generator).images[0]
        img.save(f"{out_dir}/img_{i}.jpg")

        results.append(hpsv2.score(img, d["prompt"], hps_version="v2.1")[0])
    results_df = pd.DataFrame(results, columns=["hpsv2"])
    print(results_df.mean())
    results_df.to_csv(f"{out_dir}/eval.csv", index=False)

if __name__ == "__main__":
    main()
