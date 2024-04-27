import argparse
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from diffusers import PixArtSigmaPipeline, Transformer2DModel
from tqdm import tqdm

from diffengine.evaluation import CLIPT


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the IP adapter on a set of images.")
    parser.add_argument("--model", help="Model name",
                        type=str, default="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS")
    return parser.parse_args()


def main():
    args = parse_args()

    model_name = args.model.split("/")[-1]
    out_dir = f"work_dirs/t2i_text_{model_name}"
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

    clipt = CLIPT()

    eval_ds = load_dataset("ImagenHub/Text_to_Image")["eval"]
    generator = torch.Generator(device="cuda").manual_seed(0)
    results = []
    for i, d in tqdm(enumerate(eval_ds)):
        img = pipe(d["prompt"], generator=generator).images[0]
        img.save(f"{out_dir}/img_{i}.jpg")

        results.append([d["category"], clipt(img, d["prompt"])])
    results_df = pd.DataFrame(results, columns=["category", "clipt"])
    print(results_df.mean())
    results_df.to_csv(f"{out_dir}/eval.csv", index=False)

if __name__ == "__main__":
    main()
