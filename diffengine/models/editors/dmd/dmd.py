from copy import deepcopy

import lpips
import numpy as np
import torch
import torch.nn.functional as F  # noqa
from diffusers import DDPMScheduler, PixArtAlphaPipeline
from mmengine.optim import OptimWrapperDict
from mmengine.registry import MODELS

from diffengine.datasets.utils import encode_prompt
from diffengine.models.editors.lcm.lcm_modules import (
    extract_into_tensor,
)
from diffengine.models.editors.pixart import PixArt
from diffengine.models.utils import ConstantTimeSteps


class PixArtDMD(PixArt):
    """One-step Diffusion with Distribution Matching Distillation.

    Args:
    ----
        tokenizer (dict): Config of tokenizer.
        text_encoder (dict): Config of text encoder.
        timesteps_generator (dict, optional): The timesteps generator config.
            Defaults to ``dict(type=TimeSteps, start_time_steps=950)``.
        num_ddim_timesteps (int): Number of DDIM timesteps. Defaults to 50.
        time_cond_proj_dim (int): The time condition projection dimension.
            Defaults to 256.
        cfg (float): Classifier free guidance. Defaults to 7.5.
        regression_weight (float): The regression loss weight. Defaults to 0.25.
        ema_type (str): The type of EMA.
            Defaults to 'ExponentialMovingAverage'.
        ema_momentum (float): The EMA momentum. Defaults to 0.05.

    """

    def __init__(self,  # noqa: PLR0913
                 *args,
                 tokenizer: dict,
                 text_encoder: dict,
                 vae_for_regression: dict,
                 timesteps_generator: dict | None = None,
                 num_ddim_timesteps: int = 50,
                 time_cond_proj_dim: int = 256,
                 cfg: float = 7.5,
                 regression_loss_weight: float = 0.25,
                 ema_type: str = "ExponentialMovingAverage",
                 ema_momentum: float = 0.05,
                 **kwargs) -> None:

        self.ema_cfg = dict(type=ema_type, momentum=ema_momentum)
        self.time_cond_proj_dim = time_cond_proj_dim
        self.tokenizer_config = tokenizer
        self.text_encoder_config = text_encoder
        self.vae_for_regression_config = vae_for_regression

        if timesteps_generator is None:
            timesteps_generator = {"start_time_steps": 950}

        super().__init__(
            *args,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            timesteps_generator=timesteps_generator,
            **kwargs)  # type: ignore[misc]

        if self.loss_module.use_snr:
            msg = "SNR is not supported for LCM."
            raise ValueError(msg)

        self.num_ddim_timesteps = num_ddim_timesteps
        self.cfg = cfg
        self.regression_loss_weight = regression_loss_weight

        self.lpips_loss = lpips.LPIPS(net="vgg")
        self.const_timesteps_generator = MODELS.build(
            {"type": ConstantTimeSteps})

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
        self.transformer_real = deepcopy(
            self.transformer).requires_grad_(requires_grad=False)
        self.transformer_fake = deepcopy(
            self.transformer)
        self.vae_for_regression = MODELS.build(
            self.vae_for_regression_config)

        self.register_buffer("alpha_schedule",
                             torch.sqrt(1 / self.scheduler.alphas_cumprod))
        self.register_buffer("sigma_schedule",
                             torch.sqrt(1 / self.scheduler.alphas_cumprod - 1))

        if self.pre_compute_text_embeddings:
            self.tokenizer = MODELS.build(
                self.tokenizer_config,
                default_args={"pretrained_model_name_or_path": self.model})
            self.text_encoder = MODELS.build(
                self.text_encoder_config,
                default_args={"pretrained_model_name_or_path": self.model})
        embed = encode_prompt(
            {"text": [""]},
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            caption_column="text",
            tokenizer_max_length=self.tokenizer_max_length,
        )
        if self.pre_compute_text_embeddings:
            del self.tokenizer, self.text_encoder
            torch.cuda.empty_cache()
        self.register_buffer("uncond_prompt_embeds", embed["prompt_embeds"])
        self.register_buffer("uncond_attention_mask", embed["attention_mask"].float())

        if self.gradient_checkpointing:
            self.transformer_fake.enable_gradient_checkpointing()

        super().prepare_model()

    def set_xformers(self) -> None:
        """Set xformers for model."""
        if self.enable_xformers:
            from diffusers.utils.import_utils import is_xformers_available
            if is_xformers_available():
                self.transformer.enable_xformers_memory_efficient_attention()
                self.transformer_real.enable_xformers_memory_efficient_attention()
                self.transformer_fake.enable_xformers_memory_efficient_attention()
            else:
                msg = "Please install xformers to enable memory efficient attention."
                raise ImportError(
                    msg,
                )

    @torch.no_grad()
    def infer(self,
              prompt: list[str],
              negative_prompt: str | None = None,
              height: int = 512,
              width: int = 512,
              num_inference_steps: int = 1,
              guidance_scale: float = 1.0,
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
                Defaults to 4.
            guidance_scale (float): The guidance scale. Defaults to 0.0.
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
        pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
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
                guidance_scale=guidance_scale,
                timesteps=[400],
                **kwargs).images[0]
            if output_type == "latent":
                images.append(image)
            else:
                images.append(np.array(image))

        del pipeline
        torch.cuda.empty_cache()

        return images

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

    def forward(self, *args, **kwargs) -> None:  # noqa: ARG002
        """Forward function."""
        msg = "Forward function is not implemented."
        raise NotImplementedError(msg)

    def train_step(  # noqa: PLR0915
        self, data: dict | tuple | list,
        optim_wrapper: OptimWrapperDict) -> dict[str, torch.Tensor]:
        """Train step."""
        log_vars = {}
        with optim_wrapper["transformer"].optim_context(self):
            data = self.data_preprocessor(data, training=True)
            inputs = data["inputs"]  # type: ignore[call-overload]
            num_batches = len(inputs["img"])

            latents = self._forward_vae(
                inputs["img"].to(self.weight_dtype), num_batches)

            noise = self.noise_generator(latents)

            const_timesteps = self.const_timesteps_generator(
                self.scheduler, num_batches, self.device)

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

            noise_pred = self._forward_compile(
                noise,
                encoder_attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                timestep=const_timesteps,
                added_cond_kwargs=added_cond_kwargs).chunk(2, dim=1)[0]
            pred_x_0 = self._predict_origin(
                noise_pred,
                const_timesteps,
                noise,
            )

            # distribution matching
            noise_dm = self.noise_generator(pred_x_0)
            timesteps = self.timesteps_generator(
                self.scheduler, num_batches, self.device)
            noisy_model_input, inp_noisy_latents, _ = self._preprocess_model_input(
                pred_x_0, noise_dm, timesteps)
            with torch.no_grad():
                score_real_cond = - self.transformer_real(
                    inp_noisy_latents,
                    encoder_attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timesteps,
                    added_cond_kwargs=added_cond_kwargs).sample.chunk(2, dim=1)[0]

                score_real_uncond = - self.transformer_real(
                    inp_noisy_latents,
                    encoder_attention_mask=self.uncond_attention_mask.repeat(
                        num_batches, 1, 1),
                    encoder_hidden_states=self.uncond_prompt_embeds.repeat(
                        num_batches, 1, 1),
                    timestep=timesteps,
                    added_cond_kwargs=added_cond_kwargs).sample.chunk(2, dim=1)[0]
                score_real = score_real_uncond + self.cfg * (
                    score_real_cond - score_real_uncond)

                score_fake = - self.transformer_fake(
                    inp_noisy_latents,
                    encoder_attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timesteps,
                    added_cond_kwargs=added_cond_kwargs).sample.chunk(2, dim=1)[0]
                sigmas = extract_into_tensor(self.sigma_schedule, timesteps)
                coeff = (score_fake - score_real) * sigmas

            pred_latents = self._predict_origin(
                noisy_model_input,
                const_timesteps,
                score_real,
            )
            weight = 1. / ((pred_x_0 - pred_latents
                            ).abs().mean([1, 2, 3], keepdim=True) + 1e-5
                           ).detach()
            dm_loss = F.mse_loss(pred_x_0, (pred_x_0 - weight * coeff).detach())

            # regression loss
            pred_imgs = self.vae_for_regression.decode(
                pred_x_0).sample.clamp(min=-1.0, max=1.0)
            regression_loss = self.lpips_loss(
                pred_imgs.float(),
                inputs["img"].float(),
                ).mean() * self.regression_loss_weight

            loss_dict = {}
            loss_dict["dm_loss"] = dm_loss
            loss_dict["regression_loss"] = regression_loss
            loss_dict["loss_first_stage"] = dm_loss + regression_loss

        log_vars.update(loss_dict)
        optim_wrapper["transformer"].update_params(loss_dict["loss_first_stage"])

        # train fake transformer
        with optim_wrapper["transformer_fake"].optim_context(self):
            fake_latents = pred_x_0.detach()
            fake_noise = self.noise_generator(fake_latents)
            fake_timesteps = self.timesteps_generator(
                self.scheduler, num_batches, self.device)
            (
                fake_noisy_model_input, fake_inp_noisy_latents, fake_sigmas,
            )= self._preprocess_model_input(
                fake_latents, fake_noise, fake_timesteps)
            fake_pred = self.transformer_fake(
                fake_inp_noisy_latents,
                encoder_attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                timestep=fake_timesteps,
                added_cond_kwargs=added_cond_kwargs).sample
            fake_loss = self.loss(fake_pred, fake_noise, fake_latents, fake_timesteps,
                    fake_noisy_model_input, fake_sigmas)
        log_vars.update(fake_loss)
        optim_wrapper["transformer_fake"].update_params(fake_loss["l2_loss"])

        return log_vars

    def _predict_origin(self,
                        model_output: torch.Tensor,
                        timesteps: torch.Tensor,
                        sample: torch.Tensor) -> torch.Tensor:
        """Predict the origin of the model output.

        Args:
        ----
            model_output (torch.Tensor): The model output.
            timesteps (torch.Tensor): The timesteps.
            sample (torch.Tensor): The sample.

        """
        if self.prediction_type is not None:
            # set prediction_type of scheduler if defined
            self.scheduler.register_to_config(
                prediction_type=self.prediction_type)

        sigmas = extract_into_tensor(self.sigma_schedule, timesteps)
        alphas = extract_into_tensor(self.alpha_schedule, timesteps)

        if self.scheduler.config.prediction_type == "epsilon":
            pred_x_0 = alphas * sample - sigmas * model_output
        else:
            msg = f"Prediction type {self.prediction_type} currently not supported."
            raise ValueError(msg)

        return pred_x_0
