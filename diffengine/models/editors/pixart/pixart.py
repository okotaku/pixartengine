import inspect
from copy import deepcopy
from typing import Optional, Union

import numpy as np
import torch
from diffusers import PixArtAlphaPipeline
from mmengine import print_log
from mmengine.model import BaseModel
from mmengine.registry import MODELS
from peft import get_peft_model
from torch import nn

from diffengine.models.editors.pixart.data_preprocessor import (
    DataPreprocessor,
)
from diffengine.models.losses import L2Loss
from diffengine.models.utils import TimeSteps, WhiteNoise

weight_dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

class PixArt(BaseModel):
    """PixArt.

    Args:
    ----
        tokenizer (dict): Config of tokenizer.
        scheduler (dict): Config of scheduler.
        text_encoder (dict): Config of text encoder.
        vae (dict): Config of vae.
        transformer (dict): Config of transformer.
        model (str): pretrained model name of PixArt.
            Defaults to 'PixArt-alpha/PixArt-XL-2-512x512'.
        loss (dict): Config of loss. Defaults to
            ``dict(type='L2Loss', loss_weight=1.0)``.
        transformer_lora_config (dict, optional): The LoRA config dict for
            transformer. example. dict(type=LoraConfig, r=4). Refer to the PEFT
            for more details. https://github.com/huggingface/peft
            Defaults to None.
        text_encoder_lora_config (dict, optional): The LoRA config dict for
            Text Encoder.example. dict(type=LoraConfig, r=4). Refer to the PEFT
            for more details. https://github.com/huggingface/peft
            Defaults to None.
        prediction_type (str): The prediction_type that shall be used for
            training. Choose between 'epsilon' or 'v_prediction' or leave
            `None`. If left to `None` the default prediction type of the
            scheduler will be used. Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`DataPreprocessor`.
        noise_generator (dict, optional): The noise generator config.
            Defaults to ``dict(type='WhiteNoise')``.
        timesteps_generator (dict, optional): The timesteps generator config.
            Defaults to ``dict(type='TimeSteps')``.
        input_perturbation_gamma (float): The gamma of input perturbation.
            The recommended value is 0.1 for Input Perturbation.
            Defaults to 0.0.
        vae_batch_size (int): The batch size of vae. Defaults to 8.
        weight_dtype (str): The weight dtype. Choose from "fp32", "fp16" or
            "bf16".  Defaults to 'fp32'.
        tokenizer_max_length (int): The max length of tokenizer.
            Defaults to 120.
        finetune_text_encoder (bool, optional): Whether to fine-tune text
            encoder. Defaults to False.
        gradient_checkpointing (bool): Whether or not to use gradient
            checkpointing to save memory at the expense of slower backward
            pass. Defaults to False.
        pre_compute_text_embeddings (bool): Whether or not to pre-compute text
            embeddings to save memory. Defaults to False.
        enable_xformers (bool): Whether or not to enable memory efficient
            attention. Defaults to False.
        enable_vb_loss (bool): Whether or not to enable variational bound loss.
            Defaults to False.

    """

    def __init__(  # noqa: PLR0913,C901,PLR0912,PLR0915
        self,
        tokenizer: dict,
        scheduler: dict,
        text_encoder: dict,
        vae: dict,
        transformer: dict,
        model: str = "PixArt-alpha/PixArt-XL-2-512x512",
        loss: dict | None = None,
        transformer_lora_config: dict | None = None,
        text_encoder_lora_config: dict | None = None,
        prediction_type: str | None = None,
        data_preprocessor: dict | nn.Module | None = None,
        noise_generator: dict | None = None,
        timesteps_generator: dict | None = None,
        input_perturbation_gamma: float = 0.0,
        vae_batch_size: int = 8,
        weight_dtype: str = "fp32",
        tokenizer_max_length: int = 120,
        *,
        finetune_text_encoder: bool = False,
        gradient_checkpointing: bool = False,
        pre_compute_text_embeddings: bool = False,
        enable_xformers: bool = False,
        enable_vb_loss: bool = False,
    ) -> None:
        if data_preprocessor is None:
            data_preprocessor = {"type": DataPreprocessor}
        if loss is None:
            loss = {}
        if noise_generator is None:
            noise_generator = {}
        if timesteps_generator is None:
            timesteps_generator = {}
        super().__init__(data_preprocessor=data_preprocessor)
        if (
            transformer_lora_config is not None) and (
                text_encoder_lora_config is not None) and (
                    not finetune_text_encoder):
                print_log(
                    "You are using LoRA for transformer and text encoder. "
                    "But you are not set `finetune_text_encoder=True`. "
                    "We will set `finetune_text_encoder=True` for you.")
                finetune_text_encoder = True
        if text_encoder_lora_config is not None:
            assert finetune_text_encoder, (
                "If you want to use LoRA for text encoder, "
                "you should set finetune_text_encoder=True."
            )
        if finetune_text_encoder and transformer_lora_config is not None:
            assert text_encoder_lora_config is not None, (
                "If you want to finetune text encoder with LoRA transformer, "
                "you should set text_encoder_lora_config."
            )
        if pre_compute_text_embeddings:
            assert not finetune_text_encoder

        self.model = model
        self.transformer_lora_config = deepcopy(transformer_lora_config)
        self.text_encoder_lora_config = deepcopy(text_encoder_lora_config)
        self.finetune_text_encoder = finetune_text_encoder
        self.gradient_checkpointing = gradient_checkpointing
        self.pre_compute_text_embeddings = pre_compute_text_embeddings
        self.input_perturbation_gamma = input_perturbation_gamma
        self.enable_xformers = enable_xformers
        self.vae_batch_size = vae_batch_size
        self.weight_dtype = weight_dtype_dict[weight_dtype]
        self.tokenizer_max_length = tokenizer_max_length
        self.enable_vb_loss = enable_vb_loss

        if not isinstance(loss, nn.Module):
            loss = MODELS.build(
                loss,
                default_args={"type": L2Loss, "loss_weight": 1.0})
        self.loss_module: nn.Module = loss

        assert prediction_type in [None, "epsilon", "v_prediction"]
        self.prediction_type = prediction_type

        if not self.pre_compute_text_embeddings:
            self.tokenizer = MODELS.build(
                tokenizer,
                default_args={"pretrained_model_name_or_path": model})
            self.text_encoder = MODELS.build(
                text_encoder,
                default_args={"pretrained_model_name_or_path": model})

        self.scheduler = MODELS.build(
            scheduler,
            default_args={
                "pretrained_model_name_or_path": model,
            } if not inspect.isclass(scheduler.get("type")) else None)

        self.vae = MODELS.build(
            vae,
            default_args={
                "pretrained_model_name_or_path": model,
            } if not inspect.isclass(vae.get("type")) else None)
        self.transformer = MODELS.build(
            transformer,
            default_args={
                "pretrained_model_name_or_path": model,
            } if not inspect.isclass(transformer.get("type")) else None)
        self.noise_generator = MODELS.build(
            noise_generator,
            default_args={"type": WhiteNoise})
        self.timesteps_generator = MODELS.build(
            timesteps_generator,
            default_args={"type": TimeSteps})

        if hasattr(self.vae.config, "latents_mean",
                   ) and self.vae.config.latents_mean is not None:
            self.edm_style = True
            self.register_buffer(
                "latents_mean",
                torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1))
            self.register_buffer(
                "latents_std",
                torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1))
            self.register_buffer("sigmas", self.scheduler.sigmas)
        else:
            self.edm_style = False

        self.prepare_model()
        self.set_lora()
        self.set_xformers()

        if self.weight_dtype != torch.float32:
            self.to(self.weight_dtype)

        self.register_buffer("alphas_cumprod", self.scheduler.alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev",
                             torch.cat([torch.Tensor([self.scheduler.one]),
                                        self.scheduler.alphas_cumprod[:-1]]))
        self.register_buffer("alphas", self.scheduler.alphas)
        self.register_buffer("betas", self.scheduler.betas)

    def set_lora(self) -> None:
        """Set LORA for model."""
        if self.text_encoder_lora_config is not None:
            text_encoder_lora_config = MODELS.build(
                self.text_encoder_lora_config)
            self.text_encoder = get_peft_model(
                self.text_encoder, text_encoder_lora_config)
            self.text_encoder.print_trainable_parameters()
        if self.transformer_lora_config is not None:
            transformer_lora_config = MODELS.build(
                self.transformer_lora_config)
            self.transformer = get_peft_model(self.transformer, transformer_lora_config)
            self.transformer.print_trainable_parameters()

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
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

    def set_xformers(self) -> None:
        """Set xformers for model."""
        if self.enable_xformers:
            from diffusers.utils.import_utils import is_xformers_available
            if is_xformers_available():
                self.transformer.enable_xformers_memory_efficient_attention()
            else:
                msg = "Please install xformers to enable memory efficient attention."
                raise ImportError(
                    msg,
                )

    @property
    def device(self) -> torch.device:
        """Get device information.

        Returns
        -------
            torch.device: device.

        """
        return next(self.parameters()).device

    @torch.no_grad()
    def infer(self,
              prompt: list[str],
              negative_prompt: str | None = None,
              height: int = 512,
              width: int = 512,
              num_inference_steps: int = 20,
              output_type: str = "pil",
              seed: int = 0,
              **kwargs) -> list[np.ndarray]:
        """Inference function.

        Args:
        ----
            prompt (`List[str]`):
                The prompt or prompts to guide the image generation.
            negative_prompt (`Optional[str]`):
                The prompt or prompts to guide the image generation.
                Defaults to None.
            height (int):
                The height in pixels of the generated image. Defaults to 512.
            width (int):
                The width in pixels of the generated image. Defaults to 512.
            num_inference_steps (int): Number of inference steps.
                Defaults to 20.
            output_type (str): The output format of the generate image.
                Choose between 'pil' and 'latent'. Defaults to 'pil'.
            seed (int): The seed for random number generator.
                Defaults to 0.
            **kwargs: Other arguments.

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
        images = []
        for i, p in enumerate(prompt):
            generator = torch.Generator(device=self.device).manual_seed(i + seed)
            image = pipeline(
                p,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                output_type=output_type,
                generator=generator,
                **kwargs).images[0]
            if output_type == "latent":
                images.append(image)
            else:
                images.append(np.array(image))

        del pipeline
        torch.cuda.empty_cache()

        return images

    def val_step(
            self,
            data: Union[tuple, dict, list]  # noqa
    ) -> list:
        """Val step."""
        msg = "val_step is not implemented now, please use infer."
        raise NotImplementedError(msg)

    def test_step(
            self,
            data: Union[tuple, dict, list]  # noqa
    ) -> list:
        """Test step."""
        msg = "test_step is not implemented now, please use infer."
        raise NotImplementedError(msg)

    def loss(self,
             model_pred: torch.Tensor,
             noise: torch.Tensor,
             latents: torch.Tensor,
             timesteps: torch.Tensor,
             noisy_model_input: torch.Tensor,
             sigmas: torch.Tensor | None = None,
             weight: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Calculate loss."""
        loss_dict = {}

        latent_channels = 4
        if self.transformer.config.out_channels // 2 == latent_channels:
            model_pred, model_var_values = model_pred.chunk(2, dim=1)

        if self.enable_vb_loss:
            alpha_prod_t = self.alphas_cumprod[
                timesteps][...,None,None,None]
            alpha_prod_t_prev = self.alphas_cumprod_prev[
                timesteps][...,None,None,None]
            current_beta_t = self.betas[timesteps][...,None,None,None]
            current_alpha_t = self.alphas[timesteps][...,None,None,None]
            variance = (1 - alpha_prod_t_prev
                        ) / (1 - alpha_prod_t) * current_beta_t
            variance = torch.clamp(variance, min=1e-20)

            min_log = torch.log(variance)
            max_log = torch.log(current_beta_t)
            frac = (model_var_values.float() + 1) / 2
            model_log_variance = frac * max_log  + (1 - frac) * min_log
            true_log_variance = torch.log(variance)

            pred_x0 = torch.sqrt(
                1.0 / alpha_prod_t,
            ) * noisy_model_input.float() - torch.sqrt(
                1.0 / alpha_prod_t - 1) * model_pred.float()

            posterior_mean_coef1 = (
                current_beta_t * torch.sqrt(
                    alpha_prod_t_prev) / (1.0 - alpha_prod_t)
            )
            posterior_mean_coef2 = (
                (1.0 - alpha_prod_t_prev) * torch.sqrt(
                    current_alpha_t) / (1.0 - alpha_prod_t)
            )
            model_mean = (
                posterior_mean_coef1 * pred_x0.float(
                    ) + posterior_mean_coef2 * noisy_model_input.float()
            )
            true_mean = (
                posterior_mean_coef1 * latents.float(
                    ) + posterior_mean_coef2 * noisy_model_input.float()
            )

            kl = 0.5 * (
                -1.0
                + model_log_variance
                - true_log_variance
                + torch.exp(true_log_variance - model_log_variance)
                + ((true_mean - model_mean) ** 2) * torch.exp(-model_log_variance)
            )
            kl = kl.mean(dim=[1, 2, 3]) / np.log(2.0)

            decoder_nll = - discretized_gaussian_log_likelihood(
                latents.float(), means=model_mean, log_scales=0.5 * model_log_variance,
            )
            decoder_nll = decoder_nll.mean(dim=[1, 2, 3]) / np.log(2.0)

            # At the first timestep return the decoder NLL,
            # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
            vb_loss = torch.where(timesteps == 0, decoder_nll, kl)
            loss_dict["vb_loss"] = vb_loss.mean()

        if self.edm_style:
            model_pred = self.scheduler.precondition_outputs(
                noisy_model_input, model_pred, sigmas)

        if self.prediction_type is not None:
            # set prediction_type of scheduler if defined
            self.scheduler.register_to_config(
                prediction_type=self.prediction_type)

        if self.edm_style:
            gt = latents
            msg = "EDM style is not implemented now."
            raise NotImplementedError(msg)
        if self.scheduler.config.prediction_type == "epsilon":
            gt = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            gt = self.scheduler.get_velocity(latents, noise, timesteps)
            msg = "v_prediction is not implemented now."
            raise NotImplementedError(msg)
        else:
            msg = f"Unknown prediction type {self.scheduler.config.prediction_type}"
            raise ValueError(msg)

        # calculate loss in FP32
        if self.loss_module.use_snr:
            loss = self.loss_module(
                model_pred.float(),
                gt.float(),
                timesteps,
                self.scheduler.alphas_cumprod,
                self.scheduler.config.prediction_type,
                weight=weight)
        else:
            loss = self.loss_module(
                model_pred.float(), gt.float(), weight=weight)
        loss_dict["l2_loss"] = loss
        return loss_dict

    def _preprocess_model_input(self,
                                latents: torch.Tensor,
                                noise: torch.Tensor,
                                timesteps: torch.Tensor) -> torch.Tensor:
        """Preprocess model input."""
        if self.input_perturbation_gamma > 0:
            input_noise = noise + self.input_perturbation_gamma * torch.randn_like(
                noise)
        else:
            input_noise = noise
        noisy_model_input = self.scheduler.add_noise(
            latents, input_noise, timesteps)
        if self.edm_style:
            sigmas =self._get_sigmas(timesteps)
            inp_noisy_latents = self.scheduler.precondition_inputs(
                noisy_model_input, sigmas)
        else:
            inp_noisy_latents = noisy_model_input
            sigmas = None
        return noisy_model_input, inp_noisy_latents, sigmas

    def _forward_vae(self, img: torch.Tensor, num_batches: int,
                     ) -> torch.Tensor:
        """Forward vae."""
        latents = [
            self.vae.encode(
                img[i : i + self.vae_batch_size],
            ).latent_dist.sample() for i in range(
                0, num_batches, self.vae_batch_size)
        ]
        latents = torch.cat(latents, dim=0)
        if hasattr(self, "latents_mean"):
            return (
                latents - self.latents_mean
            ) * self.vae.config.scaling_factor / self.latents_std
        return latents * self.vae.config.scaling_factor

    def _get_sigmas(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Get sigmas."""
        step_indices = [(
            self.scheduler.timesteps.to(self.device) == t
            ).nonzero().item() for t in timesteps]
        sigma = self.sigmas[step_indices].flatten()
        return sigma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    def _forward_compile(
            self,
            inp_noisy_latents: torch.Tensor,
            encoder_attention_mask: torch.Tensor | None,
            encoder_hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            added_cond_kwargs: dict) -> torch.Tensor:
        return self.transformer(
            inp_noisy_latents,
            encoder_attention_mask=encoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            added_cond_kwargs=added_cond_kwargs).sample

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

        noise = self.noise_generator(latents)

        timesteps = self.timesteps_generator(self.scheduler, num_batches,
                                            self.device)

        noisy_model_input, inp_noisy_latents, sigmas = self._preprocess_model_input(
            latents, noise, timesteps)

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
            inp_noisy_latents,
            encoder_attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timesteps,
            added_cond_kwargs=added_cond_kwargs)

        return self.loss(model_pred, noise, latents, timesteps,
                         noisy_model_input, sigmas)


def approx_standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Approximate the CDF of the standard normal distribution.

    Refert to https://github.com/PixArt-alpha/PixArt-alpha/blob/master/
    diffusion/model/diffusion_utils.py
    """
    return 0.5 * (1.0 + torch.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(
    x: torch.Tensor, means: torch.Tensor, log_scales: torch.Tensor) -> torch.Tensor:
    """Compute the log-likelihood of a Gaussian distribution discretizing.

    Refer to: https://github.com/PixArt-alpha/PixArt-alpha/blob/master/
    diffusion/model/gaussian_diffusion.py
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,  # noqa: PLR2004
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min,  # noqa: PLR2004
                    torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs
