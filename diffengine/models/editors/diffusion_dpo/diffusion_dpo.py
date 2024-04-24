from copy import deepcopy
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from diffengine.models.editors.diffusion_dpo.data_preprocessor import (
    DPODataPreprocessor,
)
from diffengine.models.editors.pixart.pixart import (
    PixArt,
    discretized_gaussian_log_likelihood,
)
from diffengine.models.losses import L2Loss


class PixArtDPO(PixArt):
    """DPO.

    Args:
    ----
        beta_dpo (int): DPO KL Divergence penalty. Defaults to 2000.
        loss (dict, optional): The loss config. Defaults to
            ``dict(type='L2Loss', loss_weight=1.0, "reduction": "none")``.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`DPODataPreprocessor`.

    """

    def __init__(self,
                 *args,
                 beta_dpo: int = 2000,
                 loss: dict | None = None,
                 data_preprocessor: dict | nn.Module | None = None,
                 **kwargs) -> None:
        if loss is None:
            loss = {"type": L2Loss, "loss_weight": 1.0,
                    "reduction": "none"}
        if data_preprocessor is None:
            data_preprocessor = {"type": DPODataPreprocessor}

        super().__init__(
            *args,
            loss=loss,
            data_preprocessor=data_preprocessor,
            **kwargs)  # type: ignore[misc]

        self.beta_dpo = beta_dpo

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
        self.orig_transformer = deepcopy(
            self.transformer).requires_grad_(requires_grad=False)

        super().prepare_model()

    def loss(  # type: ignore[override]  # noqa: PLR0915
        self,
        model_pred: torch.Tensor,
        ref_pred: torch.Tensor,
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
            ref_pred, _ = ref_pred.chunk(2, dim=1)

        if self.enable_vb_loss:
            alpha_prod_t = self.alphas_cumprod[timesteps][...,None,None,None]
            alpha_prod_t_prev = self.alphas_cumprod_prev[timesteps][...,None,None,None]
            current_beta_t = self.betas[timesteps][...,None,None,None]
            current_alpha_t = self.alphas[timesteps][...,None,None,None]
            variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
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
                current_beta_t * torch.sqrt(alpha_prod_t_prev) / (1.0 - alpha_prod_t)
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
            model_loss = self.loss_module(
                model_pred.float(),
                gt.float(),
                timesteps,
                self.scheduler.alphas_cumprod,
                self.scheduler.config.prediction_type,
                weight=weight)
            ref_loss = self.loss_module(
                ref_pred.float(),
                gt.float(),
                timesteps,
                self.scheduler.alphas_cumprod,
                self.scheduler.config.prediction_type,
                weight=weight)
        else:
            model_loss = self.loss_module(
                model_pred.float(), gt.float(), weight=weight)
            ref_loss = self.loss_module(
                ref_pred.float(), gt.float(), weight=weight)
            model_loss = model_loss.mean(
                dim=list(range(1, len(model_loss.shape))))
            ref_loss = ref_loss.mean(
                dim=list(range(1, len(ref_loss.shape))))

        model_losses_w, model_losses_l = model_loss.chunk(2)
        model_diff = model_losses_w - model_losses_l

        ref_losses_w, ref_losses_l = ref_loss.chunk(2)
        ref_diff = ref_losses_w - ref_losses_l
        scale_term = -0.5 * self.beta_dpo
        inside_term = scale_term * (model_diff - ref_diff)
        loss = -1 * F.logsigmoid(inside_term).mean()
        loss_dict["dpo_loss"] = loss
        return loss_dict

    def _forward_compile(
            self,
            inp_noisy_latents: torch.Tensor,
            encoder_attention_mask: torch.Tensor,
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

        noise = self.noise_generator(latents[:num_batches // 2])
        # repeat noise for each sample set
        noise = noise.repeat(2, 1, 1, 1)

        timesteps = self.timesteps_generator(
            self.scheduler, num_batches // 2, self.device)
        # repeat timesteps for each sample set
        timesteps = timesteps.repeat(2)

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
        # repeat text embeds for each sample set
        encoder_hidden_states = encoder_hidden_states.repeat(2, 1, 1)
        attention_mask = attention_mask.repeat(2, 1)

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
        with torch.no_grad():
            ref_pred = self.orig_transformer(
                inp_noisy_latents,
                encoder_attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timesteps,
                added_cond_kwargs=added_cond_kwargs).sample

        return self.loss(model_pred, ref_pred, noise, latents, timesteps,
                         noisy_model_input, sigmas)
