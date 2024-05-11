from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F  # noqa
from diffusers import PixArtAlphaPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import (
    retrieve_latents,
)
from diffusers.utils import load_image
from diffusers.utils.torch_utils import randn_tensor
from mmengine import print_log
from PIL import Image
from torch import nn

from diffengine.models.editors.inpaint.data_preprocessor import (
    InpaintDataPreprocessor,
)
from diffengine.models.editors.pixart import PixArt


class PixArtInpaint(PixArt):
    """Inpaint.

    Args:
    ----
        data_preprocessor (dict, optional): The pre-process config of
            :class:`InpaintDataPreprocessor`.

    """

    def __init__(self,
                 *args,
                 data_preprocessor: dict | nn.Module | None = None,
                 **kwargs) -> None:
        if data_preprocessor is None:
            data_preprocessor = {"type": InpaintDataPreprocessor}

        super().__init__(
            *args,
            data_preprocessor=data_preprocessor,
            **kwargs)  # type: ignore[misc]

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
        # Fix input channels of transformer
        in_channels = 9
        if self.transformer.in_channels != in_channels:
            out_channels = self.transformer.pos_embed.proj.out_channels
            self.transformer.register_to_config(in_channels=in_channels)

            with torch.no_grad():
                new_conv_in = nn.Conv2d(
                    in_channels, out_channels,
                    self.transformer.pos_embed.proj.kernel_size,
                    self.transformer.pos_embed.proj.stride,
                    self.transformer.pos_embed.proj.padding,
                )
                new_conv_in.weight.zero_()
                new_conv_in.weight[:, :4, :, :].copy_(
                    self.transformer.pos_embed.proj.weight)
                self.transformer.pos_embed.proj = new_conv_in

        if self.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
            if self.finetune_text_encoder:
                self.text_encoder.gradient_checkpointing_enable()

        self.vae.requires_grad_(requires_grad=False)
        print_log("Set VAE untrainable.", "current")
        if (not self.finetune_text_encoder) and (
                not self.pre_compute_text_embeddings):
            self.text_encoder.requires_grad_(requires_grad=False)
            print_log("Set Text Encoder untrainable.", "current")

    @torch.no_grad()
    def infer(self,  # noqa: PLR0913,PLR0915
              prompt: list[str],
              image: list[str | Image.Image],
              mask: list[str | Image.Image],
              negative_prompt: str | None = None,
              height: int = 512,
              width: int = 512,
              num_inference_steps: int = 20,
              guidance_scale: float = 7.5,
              output_type: str = "pil",
              resolution: list | None = None,
              aspect_ratio: list | None = None,
              seed: int = 0) -> list[np.ndarray]:
        """Inference function.

        Args:
        ----
            prompt (`List[str]`):
                The prompt or prompts to guide the image generation.
            image (`List[Union[str, Image.Image]]`):
                The image for inpainting.
            mask (`List[Union[str, Image.Image]]`):
                The mask for inpainting.
            negative_prompt (`Optional[str]`):
                The prompt or prompts to guide the image generation.
                Defaults to None.
            height (int):
                The height in pixels of the generated image. Defaults to 512.
            width (int):
                The width in pixels of the generated image. Defaults to 512.
            num_inference_steps (int): Number of inference steps.
                Defaults to 20.
            guidance_scale (float): The guidance scale for the model.
                Defaults to 7.5.
            output_type (str): The output format of the generate image.
                Choose between 'pil' and 'latent'. Defaults to 'pil'.
            resolution (list, optional): The resolution of the image.
                Defaults to None.
            aspect_ratio (list, optional): The aspect ratio of the image.
                Defaults to None.
            seed (int): The seed for random number generator.
                Defaults to 0.

        """
        if self.pre_compute_text_embeddings:
            pipeline = PixArtAlphaPipeline.from_pretrained(
                self.model,
                vae=self.vae,
                transformer=self.transformer,
                torch_dtype=self.weight_dtype,
            )
        else:
            pipeline = PixArtAlphaPipeline.from_pretrained(
                self.model,
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                transformer=self.transformer,
                torch_dtype=self.weight_dtype,
            )
        if self.prediction_type is not None:
            # set prediction_type of scheduler if defined
            scheduler_args = {"prediction_type": self.prediction_type}
            pipeline.scheduler = pipeline.scheduler.from_config(
                pipeline.scheduler.config, **scheduler_args)
        pipeline.to(self.device)
        pipeline.set_progress_bar_config(disable=True)
        added_cond_kwargs = {"resolution": resolution,
                             "aspect_ratio": aspect_ratio}
        images = []
        mask_threshold = 0.5
        for i, (p, img, m) in enumerate(zip(prompt, image, mask, strict=True)):
            generator = torch.Generator(
                device=self.device).manual_seed(i + seed)
            pil_img = load_image(img) if isinstance(img, str) else img
            pil_img = pil_img.convert("RGB").resize((width, height))
            mask_image = load_image(m) if isinstance(m, str) else m
            mask_image = mask_image.convert("L").resize((width, height))
            mask_image = (
                np.array(mask_image)[..., np.newaxis] / 255) > mask_threshold

            vae_img = pipeline.image_processor.preprocess(
                pil_img,
                height=height,
                width=width,
            )
            mask_image = torch.Tensor(mask_image).permute(
                    2, 0, 1)[None, ...]
            masked_image = vae_img * (mask_image < mask_threshold)
            masked_image = masked_image.to(self.device).to(self.weight_dtype)
            masked_image_latents = retrieve_latents(
                self.vae.encode(masked_image), generator=generator,
            ) * self.vae.config.scaling_factor
            masked_image_latents = torch.cat([masked_image_latents] * 2)
            mask = F.interpolate(
                mask_image.to(self.device).to(self.weight_dtype),
                size=(masked_image_latents.shape[2],
                        masked_image_latents.shape[3]))
            mask = torch.cat([mask] * 2)

            # Text embeddings
            text_inputs = pipeline.tokenizer(
                p,
                padding="max_length",
                max_length=self.tokenizer_max_length,
                return_tensors="pt", truncation=True)
            text_embeddings = pipeline.text_encoder(
                text_inputs.input_ids.to(self.device),
                attention_mask=text_inputs.attention_mask.to(self.device),
                ).last_hidden_state

            uncond_input = pipeline.tokenizer(
                "" if negative_prompt is None else negative_prompt,
                padding="max_length",
                max_length=self.tokenizer_max_length,
                return_tensors="pt", truncation=True)
            # Convert the text embedding back to full precision
            uncond_embeddings = pipeline.text_encoder(
                uncond_input.input_ids.to(self.device),
                attention_mask=uncond_input.attention_mask.to(self.device),
                ).last_hidden_state
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            attention_masks = torch.cat([
                uncond_input.attention_mask.to(self.device),
                text_inputs.attention_mask.to(self.device)])

            # Latent preparation
            latents = randn_tensor(
                (1, 4, height // 8, width // 8),
                generator=generator, device=self.device).to(self.weight_dtype)
            latents = latents * pipeline.scheduler.init_noise_sigma

            # Model prediction
            pipeline.scheduler.set_timesteps(num_inference_steps)
            timesteps = pipeline.scheduler.timesteps
            for t in timesteps:
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = pipeline.scheduler.scale_model_input(
                    latent_model_input, timestep=t)
                latent_model_input = torch.cat(
                    [latent_model_input, mask, masked_image_latents], dim=1)
                noise_pred = self.transformer(
                    latent_model_input,
                    encoder_attention_mask=attention_masks,
                    encoder_hidden_states=text_embeddings,
                    timestep=t.reshape(1).repeat(2).to(self.device).to(self.weight_dtype),
                    added_cond_kwargs=added_cond_kwargs).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
                noise_pred = noise_pred.chunk(2, dim=1)[0]
                latents = pipeline.scheduler.step(
                    noise_pred, t, latents, return_dict=False)[0]

            # Decoding
            latents = latents / self.vae.config.scaling_factor

            if output_type == "latent":
                images.append(latents)
            else:
                out_image = self.vae.decode(latents).sample
                out_image = (out_image / 2 + 0.5).clamp(0, 1).squeeze()
                out_npy = (
                    out_image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
                images.append(out_npy)

        del pipeline
        torch.cuda.empty_cache()

        return images

    def forward(
            self,
            inputs: dict,
            data_samples: Optional[list] = None,  # noqa
            mode: str = "loss") -> dict:
        """Forward function.

        Args:
        ----
            inputs (dict): The input dict.
            data_samples (Optional[list], optional): The data samples.
                Defaults to None.
            mode (str, optional): The mode. Defaults to "loss".

        Returns:
        -------
            dict: The loss dict.

        """
        assert mode == "loss"
        num_batches = len(inputs["img"])

        latents = self._forward_vae(inputs["img"].to(self.weight_dtype), num_batches)
        masked_latents = self._forward_vae(
            inputs["masked_image"].to(self.weight_dtype), num_batches)

        mask = F.interpolate(inputs["mask"].to(self.weight_dtype),
                             size=(latents.shape[2], latents.shape[3]))

        noise = self.noise_generator(latents)

        timesteps = self.timesteps_generator(self.scheduler, num_batches,
                                            self.device)

        noisy_model_input, inp_noisy_latents, sigmas = self._preprocess_model_input(
            latents, noise, timesteps)

        latent_model_input = torch.cat([inp_noisy_latents, mask, masked_latents], dim=1)

        if not self.pre_compute_text_embeddings:
            text_inputs = self.tokenizer(
                inputs["text"],
                max_length=self.tokenizer_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt")
            inputs["text"] = text_inputs.input_ids.to(self.device)
            attention_mask = text_inputs.attention_mask.to(self.device)
            encoder_hidden_states = self.text_encoder(
                inputs["text"], attention_mask=attention_mask)[0]
        else:
            encoder_hidden_states = inputs["prompt_embeds"].to(self.weight_dtype)
            attention_mask = inputs["attention_mask"].to(self.weight_dtype)

        if "resolution" in inputs:
            added_cond_kwargs = {"resolution": inputs["resolution"],
                                 "aspect_ratio": inputs["aspect_ratio"]}
        else:
            added_cond_kwargs = {"resolution": None, "aspect_ratio": None}

        model_pred = self._forward_compile(
            latent_model_input,
            encoder_attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timesteps,
            added_cond_kwargs=added_cond_kwargs)

        return self.loss(model_pred, noise, latents, timesteps,
                         noisy_model_input, sigmas)
