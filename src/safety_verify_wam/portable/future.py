from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from .contracts import RISK_CLASS_INDEX, SafetyBatch
from .multidomain import MultiProfilePortableSafetyCore


FUTURE_MODES = ("none", "mean", "full", "shuffled")


def _scalar_int(value: Any, *, name: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must contain one value")
        value = value.item()
    return int(value)


def extract_future_value_tokens(
    video_cache: Mapping[str, Any],
    *,
    layer_index: int = -1,
) -> torch.Tensor:
    """Return spatially preserved future V tokens from an Efficient-WAM cache.

    Efficient-WAM's multiscale cache stores condition tokens before future
    tokens.  The returned tensor is detached and flattened only across
    attention heads, retaining every future time/space position as a token.
    """

    values = video_cache.get("video_v")
    if not isinstance(values, Sequence) or not values:
        raise TypeError("video_cache must contain a non-empty video_v sequence")
    value = values[layer_index]
    if not isinstance(value, torch.Tensor):
        raise TypeError("Selected Efficient-WAM video_v entry is not a tensor")
    if value.ndim == 4:
        value = value.flatten(2)
    if value.ndim != 3:
        raise ValueError(
            "Efficient-WAM video values must be [B,N,D] or [B,N,H,Dh], "
            f"got {tuple(value.shape)}"
        )

    grid_sizes = video_cache.get("grid_sizes")
    if not isinstance(grid_sizes, Mapping):
        raise TypeError(
            "Future-only extraction requires Efficient-WAM's multiscale grid metadata"
        )
    condition_count = _scalar_int(
        grid_sizes.get("condition_seq_len"), name="condition_seq_len"
    )
    future_count = _scalar_int(
        grid_sizes.get("future_seq_len"), name="future_seq_len"
    )
    if condition_count < 0 or future_count < 1:
        raise ValueError("Invalid condition/future sequence lengths")
    if condition_count + future_count > value.shape[1]:
        raise ValueError(
            "Efficient-WAM grid metadata exceeds cached token count: "
            f"{condition_count}+{future_count}>{value.shape[1]}"
        )
    return value[:, condition_count : condition_count + future_count].detach()


@dataclass(frozen=True)
class EfficientFutureSafetyConfig:
    future_dim: int = 2048
    attention_heads: int = 4
    dropout: float = 0.0
    detach_future: bool = True

    def __post_init__(self) -> None:
        if self.future_dim < 1:
            raise ValueError("future_dim must be positive")
        if self.attention_heads < 1:
            raise ValueError("attention_heads must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EfficientFutureSafetySidecar(nn.Module):
    """Independent safety head conditioned on frozen Efficient-WAM futures.

    The existing portable safety network still encodes RGB, current robot
    state, and the candidate action chunk.  Its chunk/action features query all
    Efficient-WAM future tokens through a small cross-attention adapter.  The
    adapter output projection starts at zero, so construction preserves the
    original checkpoint logits exactly.
    """

    def __init__(
        self,
        base: MultiProfilePortableSafetyCore,
        config: EfficientFutureSafetyConfig | None = None,
        *,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.base = base
        self.config = config or EfficientFutureSafetyConfig()
        dim = int(base.config.model_dim)
        if dim % self.config.attention_heads:
            raise ValueError("Base model_dim must be divisible by attention_heads")
        self.future_norm = nn.LayerNorm(self.config.future_dim)
        self.future_projection = nn.Linear(self.config.future_dim, dim)
        self.query_norm = nn.LayerNorm(dim)
        self.future_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=self.config.attention_heads,
            dropout=self.config.dropout,
            batch_first=True,
        )
        nn.init.zeros_(self.future_attention.out_proj.weight)
        nn.init.zeros_(self.future_attention.out_proj.bias)
        self.freeze_base = bool(freeze_base)
        if self.freeze_base:
            for parameter in self.base.parameters():
                parameter.requires_grad_(False)
            self.base.eval()

    def train(self, mode: bool = True) -> "EfficientFutureSafetySidecar":
        super().train(mode)
        if self.freeze_base:
            self.base.eval()
        return self

    @staticmethod
    def _validate_future(
        future_tokens: torch.Tensor,
        future_mask: torch.Tensor | None,
        batch_size: int,
    ) -> torch.Tensor:
        if future_tokens.ndim != 3 or future_tokens.shape[0] != batch_size:
            raise ValueError(
                "future_tokens must be [B,N,D] with the safety batch size, "
                f"got {tuple(future_tokens.shape)}"
            )
        if future_mask is None:
            return torch.ones(
                future_tokens.shape[:2],
                dtype=torch.bool,
                device=future_tokens.device,
            )
        if future_mask.shape != future_tokens.shape[:2]:
            raise ValueError("future_mask shape must match future token positions")
        if future_mask.dtype != torch.bool:
            raise TypeError("future_mask must use torch.bool")
        if not bool(future_mask.any(dim=1).all()):
            raise ValueError("Every sample needs at least one future token")
        return future_mask

    @staticmethod
    def _mean_tokens(
        tokens: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = mask.to(dtype=tokens.dtype).unsqueeze(-1)
        pooled = (tokens * weights).sum(dim=1, keepdim=True) / weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        pooled_mask = torch.ones(
            (tokens.shape[0], 1), dtype=torch.bool, device=tokens.device
        )
        return pooled, pooled_mask

    def forward(
        self,
        profile_key: str,
        batch: SafetyBatch,
        future_tokens: torch.Tensor | None = None,
        *,
        future_mask: torch.Tensor | None = None,
        future_mode: str = "full",
        shuffle_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if future_mode not in FUTURE_MODES:
            raise ValueError(f"future_mode must be one of {FUTURE_MODES}")
        base_outputs = self.base(profile_key, batch)
        if future_mode == "none":
            result = dict(base_outputs)
            result["future_token_count"] = torch.zeros(
                (), dtype=torch.int64, device=batch.action.device
            )
            return result
        if future_tokens is None:
            raise ValueError(f"future_tokens are required for mode={future_mode}")
        mask = self._validate_future(
            future_tokens, future_mask, batch_size=batch.batch_size
        )
        if future_tokens.shape[-1] != self.config.future_dim:
            raise ValueError(
                f"future token dim is {future_tokens.shape[-1]}, "
                f"expected {self.config.future_dim}"
            )
        if self.config.detach_future:
            future_tokens = future_tokens.detach()
        if future_mode == "mean":
            future_tokens, mask = self._mean_tokens(future_tokens, mask)
        elif future_mode == "shuffled":
            if batch.batch_size < 2:
                raise ValueError("shuffled mode needs a batch of at least two")
            if shuffle_index is None:
                shuffle_index = torch.arange(
                    batch.batch_size - 1,
                    -1,
                    -1,
                    device=future_tokens.device,
                )
            if tuple(shuffle_index.shape) != (batch.batch_size,):
                raise ValueError("shuffle_index must be [B]")
            future_tokens = future_tokens.index_select(0, shuffle_index)
            mask = mask.index_select(0, shuffle_index)

        parameter = self.future_projection.weight
        future_tokens = future_tokens.to(
            device=parameter.device, dtype=parameter.dtype
        )
        mask = mask.to(device=parameter.device)
        queries = torch.cat(
            [
                base_outputs["risk_features"].unsqueeze(1),
                base_outputs["step_risk_features"],
            ],
            dim=1,
        ).to(dtype=parameter.dtype)
        projected = self.future_projection(self.future_norm(future_tokens))
        correction, _ = self.future_attention(
            self.query_norm(queries),
            projected,
            projected,
            key_padding_mask=~mask,
            need_weights=False,
        )
        corrected = queries + correction
        chunk_features = corrected[:, 0]
        step_features = corrected[:, 1:]
        chunk_head = (
            self.base.profile_chunk_heads[profile_key]
            if self.base.config.profile_specific_heads
            else self.base.chunk_head
        )
        step_head = (
            self.base.profile_step_heads[profile_key]
            if self.base.config.profile_specific_heads
            else self.base.step_head
        )
        class_logits = chunk_head(chunk_features)
        step_class_logits = step_head(step_features).masked_fill(
            ~batch.action_mask.unsqueeze(-1), 0.0
        )
        probabilities = torch.softmax(class_logits.float(), dim=-1)
        step_probabilities = torch.softmax(step_class_logits.float(), dim=-1)
        result = dict(base_outputs)
        result.update(
            {
                "class_logits": class_logits,
                "class_probabilities": probabilities,
                "risk_probability": probabilities[:, RISK_CLASS_INDEX],
                "step_class_logits": step_class_logits,
                "step_class_probabilities": step_probabilities,
                "step_risk_probability": step_probabilities[
                    :, :, RISK_CLASS_INDEX
                ],
                "risk_features": chunk_features,
                "step_risk_features": step_features,
                "future_token_count": torch.as_tensor(
                    future_tokens.shape[1],
                    dtype=torch.int64,
                    device=class_logits.device,
                ),
            }
        )
        return result


def future_trainable_parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
