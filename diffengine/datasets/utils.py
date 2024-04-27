# flake8: noqa: S311,ANN001
import random

import numpy as np
import torch

from diffengine.datasets.transforms import T5TextPreprocess


def encode_prompt(batch,
                text_encoder,
                tokenizer,
                caption_column,
                tokenizer_max_length,
                *,
                is_train: bool = True) -> dict[str, torch.Tensor]:
    """Encode prompt."""
    prompt_batch = batch[caption_column]
    processor = T5TextPreprocess()

    captions = []
    for caption in prompt_batch:
        if isinstance(caption, str):
            single_caption = caption
        elif isinstance(caption, list | np.ndarray):
            # take a random caption if there are multiple
            single_caption = random.choice(caption) if is_train else caption[0]
        captions.append(processor._clean_caption(single_caption))  # noqa: SLF001

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(text_encoder.device)
        attention_mask = text_inputs.attention_mask.to(text_encoder.device)
        prompt_embeds = text_encoder(
            text_input_ids, attention_mask=attention_mask,
        )[0]

    return {
        "prompt_embeds": prompt_embeds.cpu(),
        "attention_mask": attention_mask.cpu(),
    }
