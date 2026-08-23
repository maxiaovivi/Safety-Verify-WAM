from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class OVCRSConfig:
    """Compact OVCR action generator aligned with Efficient-WAM-S dimensions."""

    observation_dim: int = 48
    query_dim: int = 512
    num_queries: int = 32
    video_dim: int = 2048
    num_heads: int = 16
    head_dim: int = 128
    num_layers: int = 12
    editor_rank: int = 256
    action_dim: int = 14
    action_hidden_dim: int = 768
    action_ffn_dim: int = 3072
    action_chunk_size: int = 16
    num_registers: int = 4
    time_embedding_dim: int = 256
    num_train_timesteps: int = 1000
    distill_layers: tuple[int, ...] = (3, 6, 9, 12)
    gate_init: float = -4.0
    eps: float = 1.0e-6

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "distill_layers", tuple(int(layer) for layer in self.distill_layers)
        )
        positive_dimensions = (
            self.observation_dim,
            self.query_dim,
            self.video_dim,
            self.num_heads,
            self.head_dim,
            self.num_layers,
            self.action_dim,
            self.num_train_timesteps,
        )
        if any(dimension <= 0 for dimension in positive_dimensions):
            raise ValueError("OVCR-S dimensions and layer counts must be positive")
        if self.video_dim != self.num_heads * self.head_dim:
            raise ValueError("video_dim must equal num_heads * head_dim")
        if self.action_hidden_dim <= 0 or self.action_ffn_dim <= 0:
            raise ValueError("Action expert dimensions must be positive")
        if self.action_chunk_size <= 0 or self.num_queries <= 0:
            raise ValueError("action_chunk_size and num_queries must be positive")
        if self.editor_rank <= 0 or self.editor_rank > self.video_dim:
            raise ValueError("editor_rank must be in (0, video_dim]")
        if self.time_embedding_dim % 2:
            raise ValueError("time_embedding_dim must be even")
        invalid = [layer for layer in self.distill_layers if not 1 <= layer <= self.num_layers]
        if invalid:
            raise ValueError(f"distill_layers must use one-based indices in [1, {self.num_layers}]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        normalized = tensor.float() * torch.rsqrt(
            tensor.float().square().mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(dtype=tensor.dtype) * self.weight


class FloatLayerNorm(nn.LayerNorm):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__(dim, eps=eps, elementwise_affine=False)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return super().forward(tensor.float()).to(dtype=tensor.dtype)


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    if dim % 2:
        raise ValueError("Sinusoidal embedding dimension must be even")
    half = dim // 2
    flat_position = position.reshape(-1).to(dtype=torch.float64)
    frequencies = torch.pow(
        torch.tensor(10000.0, device=position.device, dtype=torch.float64),
        -torch.arange(half, device=position.device, dtype=torch.float64) / half,
    )
    sinusoid = torch.outer(flat_position, frequencies)
    embedding = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=-1)
    return embedding.reshape(*position.shape, dim).float()


def _fixed_position_embedding(length: int, dim: int) -> torch.Tensor:
    positions = torch.arange(length, dtype=torch.float32)
    return sinusoidal_embedding_1d(dim, positions).unsqueeze(0)


def _three_layer_mlp(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.SiLU(),
        nn.Linear(output_dim, output_dim),
        nn.SiLU(),
        nn.Linear(output_dim, output_dim),
    )


class ObservationQueryEncoder(nn.Module):
    """Turn current observation latent tokens into 32 OVCR-S queries."""

    def __init__(self, config: OVCRSConfig) -> None:
        super().__init__()
        self.config = config
        self.observation_projection = nn.Sequential(
            nn.LayerNorm(config.observation_dim, eps=config.eps),
            nn.Linear(config.observation_dim, config.query_dim),
            nn.GELU(),
            nn.Linear(config.query_dim, config.query_dim),
        )
        self.base_queries = nn.Parameter(
            torch.randn(1, 1, config.num_queries, config.query_dim)
            / math.sqrt(config.query_dim)
        )
        self.key_projection = nn.Sequential(
            nn.LayerNorm(config.query_dim, eps=config.eps),
            nn.Linear(config.query_dim, config.query_dim),
        )
        self.value_projection = nn.Sequential(
            nn.LayerNorm(config.query_dim, eps=config.eps),
            nn.Linear(config.query_dim, config.query_dim),
            nn.GELU(),
            nn.Linear(config.query_dim, config.query_dim),
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(config.query_dim, eps=config.eps),
            nn.Linear(config.query_dim, config.query_dim),
            nn.GELU(),
            nn.Linear(config.query_dim, config.query_dim),
        )

    def forward(
        self,
        observation_tokens: torch.Tensor,
        observation_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if observation_tokens.ndim == 3:
            observation_tokens = observation_tokens.unsqueeze(1)
        if observation_tokens.ndim != 4:
            raise ValueError(
                "observation_tokens must be [B,S,D] or [B,N,S,D], "
                f"got {tuple(observation_tokens.shape)}"
            )
        if observation_tokens.shape[-1] != self.config.observation_dim:
            raise ValueError(
                f"Expected observation dim {self.config.observation_dim}, "
                f"got {observation_tokens.shape[-1]}"
            )
        batch, chunks, token_count, _ = observation_tokens.shape
        if observation_mask is None:
            observation_mask = torch.ones(
                batch,
                chunks,
                token_count,
                dtype=torch.bool,
                device=observation_tokens.device,
            )
        elif observation_mask.ndim == 2:
            observation_mask = observation_mask.unsqueeze(1)
        if tuple(observation_mask.shape) != (batch, chunks, token_count):
            raise ValueError(
                "observation_mask must match [B,N,S], "
                f"got {tuple(observation_mask.shape)}"
            )
        if not observation_mask.any(dim=-1).all():
            raise ValueError("Every observation chunk needs at least one valid token")

        context = self.observation_projection(observation_tokens)
        keys = self.key_projection(context)
        values = self.value_projection(context)
        queries = self.base_queries.expand(batch, chunks, -1, -1)
        scores = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(
            self.config.query_dim
        )
        scores = scores.masked_fill(~observation_mask.unsqueeze(2), float("-inf"))
        guided_queries = torch.matmul(torch.softmax(scores, dim=-1), values)
        return self.output_projection(guided_queries)


class SharedLowRankKVEditor(nn.Module):
    """AHA-style layer editor with shared low-rank delta weights."""

    def __init__(self, config: OVCRSConfig) -> None:
        super().__init__()
        self.config = config
        self.query_projection = nn.Sequential(
            nn.LayerNorm(config.query_dim, eps=config.eps),
            nn.Linear(config.query_dim, config.video_dim),
        )
        self.layer_query_scale = nn.Parameter(torch.ones(config.num_layers, config.video_dim))
        self.layer_query_shift = nn.Parameter(torch.zeros(config.num_layers, config.video_dim))
        self.delta_norm = nn.LayerNorm(config.video_dim, eps=config.eps)
        self.delta_down = nn.Linear(config.video_dim, config.editor_rank)
        self.layer_codes = nn.Parameter(torch.zeros(config.num_layers, config.editor_rank))
        self.delta_up = nn.Linear(config.editor_rank, 2 * config.video_dim)
        self.delta_gate = nn.Parameter(
            torch.full((config.num_layers,), float(config.gate_init))
        )
        nn.init.zeros_(self.delta_up.weight)
        nn.init.zeros_(self.delta_up.bias)

    def _validate_cache(
        self,
        video_kv_cache: Sequence[Mapping[str, torch.Tensor]],
        batch_size: int,
    ) -> None:
        if len(video_kv_cache) != self.config.num_layers:
            raise ValueError(
                f"Expected {self.config.num_layers} cache layers, got {len(video_kv_cache)}"
            )
        token_count: int | None = None
        for layer, cache in enumerate(video_kv_cache, start=1):
            if "k" not in cache or "v" not in cache:
                raise ValueError(f"Cache layer {layer} must contain k and v")
            keys, values = cache["k"], cache["v"]
            if keys.ndim != 3 or values.ndim != 3 or keys.shape != values.shape:
                raise ValueError(f"Cache layer {layer} must use matching [B,S,D] tensors")
            if keys.shape[0] != batch_size or keys.shape[-1] != self.config.video_dim:
                raise ValueError(
                    f"Cache layer {layer} has incompatible shape {tuple(keys.shape)}"
                )
            if token_count is None:
                token_count = int(keys.shape[1])
            elif token_count != int(keys.shape[1]):
                raise ValueError("All cache layers must contain the same token count")

    def forward(
        self,
        queries: torch.Tensor,
        video_kv_cache: Sequence[Mapping[str, torch.Tensor]],
        *,
        trace_layers: Sequence[int] = (),
    ) -> tuple[list[dict[str, torch.Tensor]], dict[int, dict[str, torch.Tensor]]]:
        if queries.ndim != 4 or queries.shape[-1] != self.config.query_dim:
            raise ValueError("queries must be [B,N,Q,query_dim]")
        self._validate_cache(video_kv_cache, int(queries.shape[0]))
        trace_set = {int(layer) for layer in trace_layers}
        invalid = [layer for layer in trace_set if not 1 <= layer <= self.config.num_layers]
        if invalid:
            raise ValueError(f"Invalid one-based trace layers: {sorted(invalid)}")

        batch, chunks, query_count, _ = queries.shape
        updated_cache: list[dict[str, torch.Tensor]] = []
        traces: dict[int, dict[str, torch.Tensor]] = {}
        base_projected_queries = self.query_projection(queries)

        for layer_index, cache in enumerate(video_kv_cache):
            layer_number = layer_index + 1
            keys = cache["k"]
            values = cache["v"]
            token_count = int(keys.shape[1])
            projected_queries = (
                base_projected_queries * self.layer_query_scale[layer_index]
                + self.layer_query_shift[layer_index]
            )
            query_heads = projected_queries.view(
                batch,
                chunks,
                query_count,
                self.config.num_heads,
                self.config.head_dim,
            )
            key_heads = keys.view(
                batch, token_count, self.config.num_heads, self.config.head_dim
            )
            value_heads = values.view(
                batch, token_count, self.config.num_heads, self.config.head_dim
            )

            route_scores = torch.einsum("bnqhd,bshd->bhnqs", query_heads, key_heads)
            route_weights = torch.softmax(
                route_scores / math.sqrt(self.config.head_dim), dim=-1
            )
            routed = torch.einsum("bhnqs,bshd->bnqhd", route_weights, value_heads)

            decode_scores = torch.einsum("bshd,bnqhd->bhnsq", key_heads, routed)
            decode_weights = torch.softmax(
                decode_scores / math.sqrt(self.config.head_dim), dim=-1
            )
            decoded = torch.einsum(
                "bhnsq,bnqhd->bnshd", decode_weights, routed
            ).reshape(batch, chunks, token_count, self.config.video_dim)

            bottleneck = self.delta_down(self.delta_norm(decoded))
            bottleneck = F.gelu(bottleneck + self.layer_codes[layer_index])
            delta_k, delta_v = self.delta_up(bottleneck).chunk(2, dim=-1)
            gate = torch.sigmoid(self.delta_gate[layer_index]).to(
                device=delta_k.device, dtype=delta_k.dtype
            )
            effective_delta_k = gate * delta_k
            effective_delta_v = gate * delta_v
            base_k = keys.unsqueeze(1).expand(-1, chunks, -1, -1)
            base_v = values.unsqueeze(1).expand(-1, chunks, -1, -1)
            updated_cache.append(
                {
                    "k": base_k + effective_delta_k,
                    "v": base_v + effective_delta_v,
                }
            )
            if layer_number in trace_set:
                traces[layer_number] = {
                    "route_probabilities": route_weights.mean(dim=1),
                    "effective_delta_k": effective_delta_k,
                    "effective_delta_v": effective_delta_v,
                    "gate": gate.reshape(1),
                }

        return updated_cache, traces


class ActionOnlyInputEncoder(nn.Module):
    def __init__(self, config: OVCRSConfig) -> None:
        super().__init__()
        self.config = config
        self.action_encoder = _three_layer_mlp(config.action_dim, config.action_hidden_dim)
        max_length = 1 + config.action_chunk_size + config.num_registers
        self.register_buffer(
            "pos_embedding",
            _fixed_position_embedding(max_length, config.action_hidden_dim),
            persistent=False,
        )

    def forward(
        self,
        action: torch.Tensor,
        registers: torch.Tensor | None,
    ) -> torch.Tensor:
        encoded = self.action_encoder(action)
        action_length = int(action.shape[1])
        action_positions = self.pos_embedding[:, 1 : 1 + action_length].to(
            device=encoded.device, dtype=encoded.dtype
        )
        encoded = encoded + action_positions
        if registers is None:
            return encoded
        register_start = 1 + action_length
        register_positions = self.pos_embedding[
            :, register_start : register_start + registers.shape[1]
        ].to(device=registers.device, dtype=registers.dtype)
        return torch.cat([encoded, registers + register_positions], dim=1)


class CompactActionBlock(nn.Module):
    def __init__(self, config: OVCRSConfig) -> None:
        super().__init__()
        self.config = config
        self.norm1 = FloatLayerNorm(config.action_hidden_dim, config.eps)
        self.norm2 = FloatLayerNorm(config.action_hidden_dim, config.eps)
        self.wan_action_qkv = nn.Parameter(
            torch.randn(
                3,
                config.num_heads,
                config.action_hidden_dim,
                config.head_dim,
            )
            / math.sqrt(config.action_hidden_dim * config.head_dim)
        )
        self.wan_action_o = nn.Linear(
            config.video_dim, config.action_hidden_dim, bias=False
        )
        self.wan_action_norm_q = RMSNorm(config.video_dim, config.eps)
        self.wan_action_norm_k = RMSNorm(config.video_dim, config.eps)
        self.ffn = nn.Sequential(
            nn.Linear(config.action_hidden_dim, config.action_ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.action_ffn_dim, config.action_hidden_dim),
        )
        self.modulation = nn.Parameter(
            torch.randn(1, 6, config.action_hidden_dim)
            / math.sqrt(config.action_hidden_dim)
        )

    def _qkv(
        self, normalized_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = self.wan_action_qkv.to(
            device=normalized_action.device, dtype=normalized_action.dtype
        )
        query = torch.einsum("btd,hdf->bthf", normalized_action, qkv[0])
        key = torch.einsum("btd,hdf->bthf", normalized_action, qkv[1])
        value = torch.einsum("btd,hdf->bthf", normalized_action, qkv[2])
        batch, length = normalized_action.shape[:2]
        query = self.wan_action_norm_q(query.flatten(2)).view(
            batch, length, self.config.num_heads, self.config.head_dim
        )
        key = self.wan_action_norm_k(key.flatten(2)).view(
            batch, length, self.config.num_heads, self.config.head_dim
        )
        return query, key, value

    def forward(
        self,
        action_tokens: torch.Tensor,
        time_modulation: torch.Tensor,
        updated_cache: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if time_modulation.shape[:2] != action_tokens.shape[:2]:
            raise ValueError("time_modulation and action token lengths differ")
        modulation = (
            self.modulation.unsqueeze(0).to(time_modulation.dtype) + time_modulation
        ).chunk(6, dim=2)
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = [
            value.squeeze(2) for value in modulation
        ]
        normalized = self.norm1(action_tokens).float() * (1 + scale_attn) + shift_attn
        query, action_key, action_value = self._qkv(normalized.to(action_tokens.dtype))

        cache_key = updated_cache["k"]
        cache_value = updated_cache["v"]
        if cache_key.ndim != 4 or cache_key.shape[1] != 1:
            raise ValueError("OVCR-S action generation currently expects one action chunk")
        cache_key = cache_key[:, 0].view(
            action_tokens.shape[0], -1, self.config.num_heads, self.config.head_dim
        )
        cache_value = cache_value[:, 0].view_as(cache_key)
        key = torch.cat([cache_key, action_key], dim=1).transpose(1, 2)
        value = torch.cat([cache_value, action_value], dim=1).transpose(1, 2)
        response = F.scaled_dot_product_attention(
            query.transpose(1, 2), key, value, dropout_p=0.0
        ).transpose(1, 2)
        response_flat = response.flatten(2)
        projected = self.wan_action_o(
            response_flat.to(
                device=self.wan_action_o.weight.device,
                dtype=self.wan_action_o.weight.dtype,
            )
        )
        action_tokens = action_tokens + projected * gate_attn

        ffn_input = self.norm2(action_tokens).float() * (1 + scale_ffn) + shift_ffn
        ffn_weight = self.ffn[0].weight
        ffn_output = self.ffn(
            ffn_input.to(device=ffn_weight.device, dtype=ffn_weight.dtype)
        )
        action_tokens = action_tokens + ffn_output * gate_ffn
        return action_tokens, response_flat


class CompactActionDecoder(nn.Module):
    def __init__(self, config: OVCRSConfig) -> None:
        super().__init__()
        self.norm = FloatLayerNorm(config.action_hidden_dim, config.eps)
        self.action_head = nn.Sequential(
            nn.Linear(config.action_hidden_dim, config.action_dim)
        )
        self.modulation = nn.Parameter(
            torch.randn(1, 2, config.action_hidden_dim)
            / math.sqrt(config.action_hidden_dim)
        )
        nn.init.zeros_(self.action_head[0].weight)
        nn.init.zeros_(self.action_head[0].bias)

    def forward(self, tokens: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        shift, scale = (
            self.modulation.unsqueeze(0).to(time_embedding.dtype)
            + time_embedding.unsqueeze(2)
        ).chunk(2, dim=2)
        decoded = self.norm(tokens) * (1 + scale.squeeze(2)) + shift.squeeze(2)
        weight = self.action_head[0].weight
        return self.action_head(decoded.to(device=weight.device, dtype=weight.dtype))


class CompactActionExpert(nn.Module):
    """Efficient-WAM-S-sized action branch with compatible parameter names."""

    def __init__(self, config: OVCRSConfig) -> None:
        super().__init__()
        self.config = config
        self.input_encoder = ActionOnlyInputEncoder(config)
        self.time_embedding = nn.Sequential(
            nn.Linear(config.time_embedding_dim, config.action_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.action_hidden_dim, config.action_hidden_dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.action_hidden_dim, 6 * config.action_hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [CompactActionBlock(config) for _ in range(config.num_layers)]
        )
        self.registers = (
            nn.Parameter(
                torch.empty(1, config.num_registers, config.action_hidden_dim).normal_(
                    std=0.02
                )
            )
            if config.num_registers > 0
            else None
        )
        self.decoder = CompactActionDecoder(config)
        self._initialize_linears()

    def _initialize_linears(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.decoder.action_head[0].weight)
        nn.init.zeros_(self.decoder.action_head[0].bias)
        for module in self.time_embedding.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)


class OVCRSActionGenerator(nn.Module):
    """Generate one action chunk from current observation and compact video K/V."""

    def __init__(self, config: OVCRSConfig) -> None:
        super().__init__()
        self.config = config
        self.query_encoder = ObservationQueryEncoder(config)
        self.cache_editor = SharedLowRankKVEditor(config)
        self.action_expert = CompactActionExpert(config)

    def _expand_action_time(
        self,
        action_t: torch.Tensor,
        action_length: int,
        full_length: int,
    ) -> torch.Tensor:
        if action_t.ndim == 0:
            action_t = action_t.reshape(1)
        if action_t.ndim == 1:
            expanded = action_t.unsqueeze(1).expand(-1, full_length)
        elif action_t.ndim == 2 and action_t.shape[1] == 1:
            expanded = action_t.expand(-1, full_length)
        elif action_t.ndim == 2 and action_t.shape[1] == action_length:
            if full_length == action_length:
                expanded = action_t
            else:
                register_time = action_t[:, -1:].expand(-1, full_length - action_length)
                expanded = torch.cat([action_t, register_time], dim=1)
        else:
            raise ValueError(
                "action_t must be scalar, [B], [B,1], or [B,action_chunk_size]"
            )
        return expanded

    def _time_conditioning(
        self,
        action_t: torch.Tensor,
        action_length: int,
        full_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expanded = self._expand_action_time(action_t, action_length, full_length)
        embedding = sinusoidal_embedding_1d(
            self.config.time_embedding_dim, expanded
        ).to(device=expanded.device)
        time_weight = self.action_expert.time_embedding[0].weight
        embedding = self.action_expert.time_embedding(
            embedding.to(device=time_weight.device, dtype=time_weight.dtype)
        )
        projection = self.action_expert.time_projection(embedding).view(
            embedding.shape[0], full_length, 6, self.config.action_hidden_dim
        )
        return embedding, projection

    def prepare_conditioning(
        self,
        observation_tokens: torch.Tensor,
        video_kv_cache: Sequence[Mapping[str, torch.Tensor]],
        observation_mask: torch.Tensor | None = None,
        *,
        return_trace: bool = False,
    ) -> dict[str, Any]:
        queries = self.query_encoder(observation_tokens, observation_mask)
        if queries.shape[1] != 1:
            raise ValueError(
                "OVCR-S action generation currently accepts one observation/action chunk"
            )
        trace_layers = self.config.distill_layers if return_trace else ()
        updated_cache, editor_trace = self.cache_editor(
            queries, video_kv_cache, trace_layers=trace_layers
        )
        return {
            "queries": queries,
            "updated_cache": updated_cache,
            "editor_trace": editor_trace,
        }

    def predict_velocity(
        self,
        noisy_action: torch.Tensor,
        action_t: torch.Tensor,
        conditioning: Mapping[str, Any],
        *,
        return_trace: bool = False,
    ) -> dict[str, Any]:
        if noisy_action.ndim != 3:
            raise ValueError("noisy_action must be [B,T,A]")
        expected = (self.config.action_chunk_size, self.config.action_dim)
        if tuple(noisy_action.shape[1:]) != expected:
            raise ValueError(
                f"Expected noisy_action [B,{expected[0]},{expected[1]}], "
                f"got {tuple(noisy_action.shape)}"
            )
        registers = (
            self.action_expert.registers.expand(noisy_action.shape[0], -1, -1)
            if self.action_expert.registers is not None
            else None
        )
        action_tokens = self.action_expert.input_encoder(noisy_action, registers)
        action_length = int(noisy_action.shape[1])
        time_embedding, time_modulation = self._time_conditioning(
            action_t, action_length, int(action_tokens.shape[1])
        )
        responses: dict[int, torch.Tensor] = {}
        trace_set = set(self.config.distill_layers) if return_trace else set()
        updated_cache = conditioning["updated_cache"]
        if len(updated_cache) != self.config.num_layers:
            raise ValueError("Prepared conditioning has an invalid layer count")
        for layer_index, block in enumerate(self.action_expert.blocks):
            action_tokens, response = block(
                action_tokens, time_modulation, updated_cache[layer_index]
            )
            layer_number = layer_index + 1
            if layer_number in trace_set:
                responses[layer_number] = response[:, :action_length]
        prediction = self.action_expert.decoder(action_tokens, time_embedding)
        outputs: dict[str, Any] = {
            "action_velocity": prediction[:, :action_length],
            "action_hidden": action_tokens[:, :action_length],
        }
        if return_trace:
            outputs["action_responses"] = responses
        return outputs

    def forward(
        self,
        noisy_action: torch.Tensor,
        action_t: torch.Tensor,
        observation_tokens: torch.Tensor,
        video_kv_cache: Sequence[Mapping[str, torch.Tensor]],
        observation_mask: torch.Tensor | None = None,
        *,
        return_trace: bool = False,
    ) -> dict[str, Any]:
        conditioning = self.prepare_conditioning(
            observation_tokens,
            video_kv_cache,
            observation_mask,
            return_trace=return_trace,
        )
        outputs = self.predict_velocity(
            noisy_action, action_t, conditioning, return_trace=return_trace
        )
        outputs["queries"] = conditioning["queries"]
        if return_trace:
            outputs["editor_trace"] = conditioning["editor_trace"]
        return outputs

    @torch.inference_mode()
    def generate(
        self,
        observation_tokens: torch.Tensor,
        video_kv_cache: Sequence[Mapping[str, torch.Tensor]],
        observation_mask: torch.Tensor | None = None,
        *,
        num_steps: int = 4,
        flow_shift: float = 5.0,
        initial_noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        batch_size = int(observation_tokens.shape[0])
        parameter = next(self.parameters())
        if initial_noise is None:
            action = torch.randn(
                batch_size,
                self.config.action_chunk_size,
                self.config.action_dim,
                device=parameter.device,
                dtype=parameter.dtype,
                generator=generator,
            )
        else:
            action = initial_noise.to(device=parameter.device, dtype=parameter.dtype)
        conditioning = self.prepare_conditioning(
            observation_tokens,
            video_kv_cache,
            observation_mask,
            return_trace=False,
        )
        linear_sigma = torch.linspace(
            1.0, 0.0, num_steps + 1, device=parameter.device, dtype=torch.float32
        )
        sigma = flow_shift * linear_sigma / (1 + (flow_shift - 1) * linear_sigma)
        for current_sigma, next_sigma in zip(sigma[:-1], sigma[1:]):
            action_t = (
                current_sigma * self.config.num_train_timesteps
            ).expand(batch_size).to(dtype=parameter.dtype)
            velocity = self.predict_velocity(
                action, action_t, conditioning, return_trace=False
            )["action_velocity"]
            action = action + velocity * (next_sigma - current_sigma).to(action.dtype)
        return action

    def load_efficient_action_expert(
        self,
        checkpoint: Mapping[str, Any],
        *,
        strict: bool = True,
    ) -> tuple[list[str], list[str]]:
        """Initialize matching action tensors from an Efficient-WAM checkpoint."""

        source: Mapping[str, Any] = checkpoint
        for container_key in ("model", "state_dict"):
            candidate = source.get(container_key) if isinstance(source, Mapping) else None
            if isinstance(candidate, Mapping):
                source = candidate
                break
        tensor_source = {
            str(key): value for key, value in source.items() if isinstance(value, torch.Tensor)
        }
        target_state = self.action_expert.state_dict()
        loaded: dict[str, torch.Tensor] = {}
        mismatched: list[str] = []
        for target_key, target_value in target_state.items():
            suffix = f"action_expert.{target_key}"
            matches = [
                value
                for key, value in tensor_source.items()
                if key == target_key or key.endswith(suffix)
            ]
            if len(matches) != 1:
                continue
            if tuple(matches[0].shape) != tuple(target_value.shape):
                mismatched.append(target_key)
                continue
            loaded[target_key] = matches[0]
        missing = sorted(set(target_state) - set(loaded))
        if strict and (missing or mismatched):
            raise RuntimeError(
                "Efficient action expert is incompatible with OVCR-S: "
                f"missing={missing[:10]}, mismatched={mismatched[:10]}"
            )
        self.action_expert.load_state_dict(loaded, strict=False)
        return missing, mismatched
