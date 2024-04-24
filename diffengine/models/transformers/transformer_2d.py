import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward, _chunked_feed_forward
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import PatchEmbed, PixArtAlphaTextProjection
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormSingle
from diffusers.models.transformers.transformer_2d import (
    Transformer2DModelOutput,
)
from torch import nn

from diffengine.models.layers import get_layer_norm


class BasicTransformerBlock(nn.Module):
    """A basic Transformer block.

    Refer to diffusers/src/diffusers/models/attention.py
    """

    def __init__(  # noqa: PLR0913
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: int | None = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: int | None = None,
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        attention_type: str = "default",
        ff_inner_dim: int | None = None,
        *,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        final_dropout: bool = False,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ) -> None:
        super().__init__()
        assert norm_type == "ada_norm_single"
        self.only_cross_attention = only_cross_attention

        self.norm_type = norm_type
        self.num_embeds_ada_norm = num_embeds_ada_norm

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        self.norm1 = get_layer_norm(
            dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=(
                cross_attention_dim if only_cross_attention else None),
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        # 2. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            self.norm2 = get_layer_norm(
                dim, norm_eps, elementwise_affine=norm_elementwise_affine)

            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=(
                    cross_attention_dim if not double_self_attention else None
                ),
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm2 = None
            self.attn2 = None

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        # 4. Fuser
        assert attention_type not in ["gated", "gated-text-image"]

        # 5. Scale-shift for PixArt-Alpha.
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

        # let chunk size default to None
        self._chunk_size: int | None = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self,
                               chunk_size: int | None,
                               dim: int = 0) -> None:
        """Set chunk parameters for feed forward."""
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        timestep: torch.LongTensor | None = None,
        cross_attention_kwargs: dict | None = None,
        class_labels: torch.LongTensor | None = None,  # noqa: ARG002
        added_cond_kwargs: dict[str, torch.Tensor] | None = None,  # noqa: ARG002
    ) -> torch.FloatTensor:
        """Forward function for the basic transformer block."""
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        # ada_norm_single
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        norm_hidden_states = norm_hidden_states.squeeze(1)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=(
                encoder_hidden_states if self.only_cross_attention else None),
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        # ada_norm_single
        attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:  # noqa: PLR2004
            hidden_states = hidden_states.squeeze(1)

        # 3. Cross-Attention
        if self.attn2 is not None:
            # For PixArt norm2 isn't applied here:
            # https://github.com/PixArt-alpha/PixArt-alpha/blob/
            # 0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/
            # model/nets/PixArtMS.py#L70C1-L76C103
            norm_hidden_states = hidden_states

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        # ada_norm_single
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(
                self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        # ada_norm_single
        ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:  # noqa: PLR2004
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class Transformer2DModel(ModelMixin, ConfigMixin):
    """Transformer2DModel.

    Refer to diffusers/src/diffusers/models/transformers/transformer_2d.py

    The difference from original source is:
        - Support only PixArt blocks
        - Support apex

    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(  # noqa: PLR0913
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: int | None = None,
        out_channels: int | None = None,
        num_layers: int = 1,  # noqa: ARG002
        dropout: float = 0.0,  # noqa: ARG002
        norm_num_groups: int = 32,  # noqa: ARG002
        cross_attention_dim: int | None = None,  # noqa: ARG002
        sample_size: int | None = None,  # noqa: ARG002
        num_vector_embeds: int | None = None,
        patch_size: int | None = None,
        activation_fn: str = "geglu",  # noqa: ARG002
        num_embeds_ada_norm: int | None = None,  # noqa: ARG002
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,  # noqa: ARG002
        attention_type: str = "default",  # noqa: ARG002
        caption_channels: int | None = None,
        interpolation_scale: float | None = None,
        *,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,  # noqa: ARG002
        double_self_attention: bool = False,  # noqa: ARG002
        upcast_attention: bool = False,  # noqa: ARG002
        norm_elementwise_affine: bool = True,  # noqa: ARG002
        attention_bias: bool = False,  # noqa: ARG002
    ) -> None:
        super().__init__()
        assert use_linear_projection is False
        assert interpolation_scale is None
        assert num_vector_embeds is None

        # check whether the _init_patched_inputs or not
        assert in_channels is not None
        assert patch_size is not None

        self.caption_channels = caption_channels
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.inner_dim = (
            self.config.num_attention_heads * self.config.attention_head_dim)
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.gradient_checkpointing = False

        self._init_patched_inputs(norm_type=norm_type)

    def _init_patched_inputs(self, norm_type: str) -> None:
        """Initialize model parameters for patched inputs."""
        assert self.config.sample_size is not None, (
            "Transformer2DModel over patched input must provide sample_size")

        self.height = self.config.sample_size
        self.width = self.config.sample_size

        self.patch_size = self.config.patch_size
        interpolation_scale = (
            self.config.interpolation_scale
            if self.config.interpolation_scale is not None
            else max(self.config.sample_size // 64, 1)
        )
        self.pos_embed = PatchEmbed(
            height=self.config.sample_size,
            width=self.config.sample_size,
            patch_size=self.config.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.inner_dim,
            interpolation_scale=interpolation_scale,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    cross_attention_dim=self.config.cross_attention_dim,
                    activation_fn=self.config.activation_fn,
                    num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                    attention_bias=self.config.attention_bias,
                    only_cross_attention=self.config.only_cross_attention,
                    double_self_attention=self.config.double_self_attention,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    attention_type=self.config.attention_type,
                )
                for _ in range(self.config.num_layers)
            ],
        )

        assert self.config.norm_type == "ada_norm_single"
        self.norm_out = get_layer_norm(
            self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, self.inner_dim) / self.inner_dim**0.5)
        self.proj_out = nn.Linear(
            self.inner_dim,
            self.config.patch_size * self.config.patch_size * self.out_channels,
        )

        # PixArt-Alpha blocks.
        self.use_additional_conditions = self.config.sample_size == 128  # noqa: PLR2004
        self.adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            use_additional_conditions=self.use_additional_conditions,
        )

        assert self.caption_channels is not None
        self.caption_projection = PixArtAlphaTextProjection(
            in_features=self.caption_channels, hidden_size=self.inner_dim,
        )

    def _set_gradient_checkpointing(self, module: nn.Module,
                                    *,
                                    value: bool = False) -> None:
        """Set gradient checkpointing for the module."""
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        timestep: torch.LongTensor | None = None,
        added_cond_kwargs: dict[str, torch.Tensor] | None = None,
        class_labels: torch.LongTensor | None = None,
        cross_attention_kwargs: dict | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        *,
        return_dict: bool = True,
    ) -> tuple | Transformer2DModelOutput:
        """Forward function for the Transformer2DModel."""
        if attention_mask is not None and attention_mask.ndim == 2:  # noqa: PLR2004
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and (
            encoder_attention_mask.ndim == 2):  # noqa: PLR2004
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        height, width = (
            hidden_states.shape[-2] // self.patch_size,
            hidden_states.shape[-1] // self.patch_size,
        )
        (
            hidden_states, encoder_hidden_states, timestep, embedded_timestep,
        ) = self._operate_on_patched_inputs(
            hidden_states, encoder_hidden_states, timestep, added_cond_kwargs,
        )

        # 2. Blocks
        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(  # noqa
                    module: nn.Module, return_dict: dict | None = None):
                    def custom_forward(*inputs):  # noqa
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                )

        # 3. Output
        output = self._get_output_for_patched_inputs(
            hidden_states=hidden_states,
            timestep=timestep,
            class_labels=class_labels,
            embedded_timestep=embedded_timestep,
            height=height,
            width=width,
        )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def _operate_on_patched_inputs(self,
                                   hidden_states: torch.Tensor,
                                   encoder_hidden_states: torch.Tensor,
                                   timestep: torch.Tensor,
                                   added_cond_kwargs: dict) -> tuple:
        """Operate on patched inputs."""
        batch_size = hidden_states.shape[0]
        hidden_states = self.pos_embed(hidden_states)
        embedded_timestep = None

        if self.use_additional_conditions and added_cond_kwargs is None:
            msg = ("`added_cond_kwargs` cannot be None when using additional"
                   " conditions for `adaln_single`.")
            raise ValueError(msg)

        timestep, embedded_timestep = self.adaln_single(
            timestep, added_cond_kwargs,
            batch_size=batch_size, hidden_dtype=hidden_states.dtype,
        )

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(
            batch_size, -1, hidden_states.shape[-1])

        return hidden_states, encoder_hidden_states, timestep, embedded_timestep

    def _get_output_for_patched_inputs(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,  # noqa: ARG002
        class_labels: torch.Tensor,  # noqa: ARG002
        embedded_timestep: torch.Tensor,
        height: int | None = None,
        width: int | None = None,
    ) -> torch.Tensor:
        """Get output for patched inputs."""
        shift, scale = (
            self.scale_shift_table[None] + embedded_timestep[:, None]
        ).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)
        # Modulation
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.squeeze(1)

        # unpatchify
        hidden_states = hidden_states.reshape(
            shape=(-1, height, width, self.patch_size, self.patch_size,
                   self.out_channels),
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        return hidden_states.reshape(
            shape=(-1, self.out_channels, height * self.patch_size,
                   width * self.patch_size),
        )
