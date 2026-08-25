from __future__ import annotations

import importlib
import logging
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn

from ..config import resolve_project_path
from .safety_verifier import RiskHeadConfig, SafetyRiskHead, SafetyVerifyWAM


LOGGER = logging.getLogger(__name__)
EXPECTED_EFFICIENT_WAM_COMMIT = "2bd75a8c56acfcd5754b98c7ed313176911ccae0"


def _trusted_torch_load(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a dictionary checkpoint: {path}")
    return payload


def _verify_external_source(source_root: Path, expected_commit: str) -> None:
    required = (
        source_root / "models" / "small_wam.py",
        source_root / "models" / "compact_wan.py",
        source_root / "third_party" / "wan" / "utils" / "fm.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Efficient-WAM source is incomplete. Provide the pinned external checkout. "
            f"Missing: {missing}"
        )
    try:
        actual = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Cannot verify Efficient-WAM checkout at {source_root}") from exc
    if actual != expected_commit:
        raise RuntimeError(
            "Efficient-WAM source revision mismatch: "
            f"expected {expected_commit}, got {actual}."
        )


def _import_external_modules(source_root: Path) -> dict[str, Any]:
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    importlib.invalidate_caches()
    compact_module = importlib.import_module("models.compact_wan")
    small_module = importlib.import_module("models.small_wam")
    scheduler_module = importlib.import_module("third_party.wan.utils.fm")
    return {
        "CompactWANConfig": compact_module.CompactWANConfig,
        "CompactWANModel": compact_module.CompactWANModel,
        "SmallWAMActionConfig": small_module.SmallWAMActionConfig,
        "SmallWAMActionModel": small_module.SmallWAMActionModel,
        "FlowMatchScheduler": scheduler_module.FlowMatchScheduler,
    }


def _strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state and all(key.startswith("module.") for key in state):
        return {key[len("module.") :]: value for key, value in state.items()}
    return state


class EfficientWAMSafetyBackbone(nn.Module):
    """Pinned Efficient-WAM adapter for candidate-conditioned imagination.

    The external model remains unmodified. Current robot state, candidate
    actions, and task text enter the same state/action and video paths used by
    the original Efficient-WAM checkpoint. The candidate action is clamped at
    the clean flow endpoint while video latents are denoised. Late frozen
    ActionExpert streams are exposed to a low-rank three-class safety reader.
    """

    def __init__(
        self,
        wam: nn.Module,
        scheduler_class: type,
        *,
        num_video_frames: int,
        rollout_steps: int,
        flow_shift: float,
        rollout_seed: int,
        randomize_training_noise: bool,
        safety_tap_layers: int,
    ) -> None:
        super().__init__()
        self.wam = wam
        self.scheduler_class = scheduler_class
        self.num_video_frames = int(num_video_frames)
        self.rollout_steps = int(rollout_steps)
        self.flow_shift = float(flow_shift)
        self.rollout_seed = int(rollout_seed)
        self.randomize_training_noise = bool(randomize_training_noise)
        self.safety_tap_layers = int(safety_tap_layers)
        self.action_feature_dim = int(wam.config.ae_dim)
        self.action_dim = int(wam.config.action_dim)
        self.action_steps = int(wam.config.chunk_size)
        self._has_trainable_wam_parameters = False
        if not wam.compact_wan.is_multiscale:
            raise ValueError(
                "Safety verification requires an Efficient-WAM checkpoint with "
                "compact_wan.future_video_size configured"
            )
        if self.rollout_steps < 1:
            raise ValueError("rollout_steps must be at least 1")
        if self.num_video_frames < 4 or self.num_video_frames % 4 != 0:
            raise ValueError("num_video_frames must be a positive multiple of 4")
        if not 1 <= self.safety_tap_layers <= int(wam.config.ae_num_layers):
            raise ValueError(
                "safety_tap_layers must be in [1, Efficient-WAM action layers]"
            )
        self.wam.configure_teacache(enabled=False)

    def train(self, mode: bool = True) -> "EfficientWAMSafetyBackbone":
        super().train(mode)
        # Efficient-WAM is a frozen feature generator. Keep its stochastic/stateful
        # behavior in evaluation mode while the low-rank safety reader trains.
        self.wam.eval()
        return self

    def configure_trainability(self, mode: str) -> None:
        for parameter in self.wam.parameters():
            parameter.requires_grad_(False)
        mode = str(mode)
        if mode != "head_only":
            raise ValueError(
                "Safety-LoRA keeps the complete Efficient-WAM frozen; "
                f"backbone.train_mode must be 'head_only', got {mode!r}"
            )
        vae = getattr(self.wam.compact_wan.video_model, "vae", None)
        if vae is not None and hasattr(vae, "parameters"):
            for parameter in vae.parameters():
                parameter.requires_grad_(False)
        self._has_trainable_wam_parameters = False
        self.wam.eval()

    def _encode_state_and_actions(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        registers: torch.Tensor | None,
    ) -> torch.Tensor:
        encoder = self.wam.action_expert.input_encoder
        if not hasattr(encoder, "state_encoder") or not hasattr(
            encoder, "action_encoder"
        ):
            raise RuntimeError("Pinned Efficient-WAM state/action encoder changed")
        return encoder(state.unsqueeze(1), action, registers)

    def _prepare_text_embeddings(
        self,
        text_embeddings: torch.Tensor | Sequence[torch.Tensor],
        batch_size: int,
    ) -> list[torch.Tensor]:
        text_dim = int(self.wam.compact_wan.video_model.wan_model.text_dim)
        if isinstance(text_embeddings, torch.Tensor):
            if text_embeddings.ndim == 2 and batch_size == 1:
                items = [text_embeddings]
            elif text_embeddings.ndim == 3 and text_embeddings.shape[0] == batch_size:
                items = list(text_embeddings.unbind(0))
            else:
                raise ValueError(
                    "text_embeddings must be [B,L,D], or [L,D] for batch size 1"
                )
        else:
            items = list(text_embeddings)
        if len(items) != batch_size:
            raise ValueError(
                f"Expected {batch_size} text embeddings, received {len(items)}"
            )
        for item in items:
            if item.ndim != 2 or item.shape[-1] != text_dim:
                raise ValueError(
                    f"Each text embedding must be [L,{text_dim}], got {tuple(item.shape)}"
                )
            if not torch.isfinite(item).all():
                raise ValueError("Text embedding contains NaN or infinity")
        return items

    def _joint_forward(
        self,
        condition_latent: torch.Tensor,
        future_latent: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        text_embeddings: torch.Tensor | Sequence[torch.Tensor],
        video_t: torch.Tensor,
        *,
        return_features: bool,
    ) -> dict[str, torch.Tensor]:
        wam = self.wam
        video_tokens, seq_lens, layout, freqs = wam.compact_wan.prepare_multiscale_video_tokens(
            condition_latent, future_latent
        )
        text_batch = self._prepare_text_embeddings(
            text_embeddings,
            int(action.shape[0]),
        )
        text_context = wam.compact_wan.prepare_text_context(text_batch)
        video_head_time_emb, video_adaln_params = wam._build_video_time_embeddings(
            video_t, video_tokens.shape[1]
        )

        registers = None
        if wam.action_expert.config.num_registers > 0 and wam.action_expert.registers is not None:
            registers = wam.action_expert.registers.expand(action.shape[0], -1, -1)
        action_tokens = self._encode_state_and_actions(state, action, registers)
        action_t = torch.zeros(
            action.shape[0], device=action.device, dtype=action.dtype
        )
        _, action_adaln_params = wam._build_action_time_embeddings(
            action_t, action_tokens.shape[1]
        )

        action_feature_taps: list[torch.Tensor] = []
        first_tap_layer = int(wam.config.ae_num_layers) - self.safety_tap_layers
        for layer_idx in range(wam.config.ae_num_layers):
            wan_layer = wam.compact_wan.video_model.wan_model.blocks[layer_idx]
            action_block = wam.action_expert.blocks[layer_idx]
            video_modulation = wam._block_modulation(wan_layer, video_adaln_params)
            action_modulation = wam._block_modulation(action_block, action_adaln_params)
            video_tokens, action_tokens = wam._joint_attention(
                video_tokens,
                action_tokens,
                video_modulation,
                action_modulation,
                layer_idx,
                seq_lens,
                layout,
                freqs,
            )

            cross_dtype = wam.compact_wan.video_model.precision
            with torch.autocast("cuda", dtype=cross_dtype, enabled=video_tokens.is_cuda):
                cross_out = wan_layer.cross_attn(
                    wan_layer.norm3(video_tokens), text_context, None
                )
            video_tokens = video_tokens + cross_out
            video_ffn_in = (
                wan_layer.norm2(video_tokens).float()
                * (1 + video_modulation[4].squeeze(2))
                + video_modulation[3].squeeze(2)
            )
            video_ffn_weight = wan_layer.ffn[0].weight
            video_ffn = wan_layer.ffn(
                video_ffn_in.to(
                    device=video_ffn_weight.device, dtype=video_ffn_weight.dtype
                )
            )
            with torch.amp.autocast(
                "cuda", dtype=torch.float32, enabled=video_tokens.is_cuda
            ):
                video_tokens = (
                    video_tokens + video_ffn * video_modulation[5].squeeze(2)
                )

            action_ffn_in = (
                action_block.norm2(action_tokens).float()
                * (1 + action_modulation[4].squeeze(2))
                + action_modulation[3].squeeze(2)
            )
            action_ffn_weight = action_block.ffn[0].weight
            action_ffn = action_block.ffn(
                action_ffn_in.to(
                    device=action_ffn_weight.device, dtype=action_ffn_weight.dtype
                )
            )
            with torch.amp.autocast(
                "cuda", dtype=torch.float32, enabled=action_tokens.is_cuda
            ):
                action_tokens = (
                    action_tokens + action_ffn * action_modulation[5].squeeze(2)
                )
            if return_features and layer_idx >= first_tap_layer:
                action_feature_taps.append(action_tokens)

        if not return_features:
            return {
                "video_pred": wam.compact_wan.apply_multiscale_video_head(
                    video_tokens, video_head_time_emb, layout
                )
            }

        if len(action_feature_taps) != self.safety_tap_layers:
            raise RuntimeError(
                "Efficient-WAM returned an incomplete set of safety taps: "
                f"{len(action_feature_taps)} != {self.safety_tap_layers}"
            )
        action_steps = int(action.shape[1])
        register_start = 1 + action_steps
        tapped_tokens = torch.stack(action_feature_taps, dim=1)
        if tapped_tokens.shape[2] < register_start:
            raise RuntimeError(
                "ActionExpert token sequence is shorter than state + action chunk"
            )
        return {
            "state_feature_taps": tapped_tokens[:, :, :1],
            "action_feature_taps": tapped_tokens[:, :, 1:register_start],
            "register_feature_taps": tapped_tokens[:, :, register_start:],
        }

    def _make_future_noise(
        self,
        condition_latent: torch.Tensor,
        *,
        training: bool,
    ) -> torch.Tensor:
        future_size = self.wam.compact_wan.config.future_video_size
        if future_size is None:
            raise RuntimeError("Efficient-WAM multiscale future size is missing")
        future_h, future_w = (int(value) for value in future_size)
        if future_h % 16 != 0 or future_w % 16 != 0:
            raise ValueError("future_video_size must be divisible by the VAE spatial factor 16")
        shape = (
            condition_latent.shape[0],
            condition_latent.shape[1],
            self.num_video_frames // 4,
            future_h // 16,
            future_w // 16,
        )
        generator = None
        if not (training and self.randomize_training_noise):
            generator = torch.Generator(device=condition_latent.device)
            generator.manual_seed(self.rollout_seed)
        return torch.randn(
            shape,
            device=condition_latent.device,
            dtype=condition_latent.dtype,
            generator=generator,
        )

    def imagine(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        text_embeddings: torch.Tensor | Sequence[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if state.shape[1:] != (self.wam.config.state_dim,):
            raise ValueError(
                f"Expected state [B,{self.wam.config.state_dim}], "
                f"got {tuple(state.shape)}"
            )
        if action.shape[1:] != (self.action_steps, self.action_dim):
            raise ValueError(
                f"Expected action [B, {self.action_steps}, {self.action_dim}], "
                f"got {tuple(action.shape)}"
            )
        if state.shape[0] != action.shape[0] or image.shape[0] != action.shape[0]:
            raise ValueError("Image, state, and action batch sizes differ")
        model_parameter = next(self.wam.parameters())
        device = model_parameter.device
        video_dtype = self.wam.compact_wan.video_model.precision
        action_dtype = next(self.wam.action_expert.parameters()).dtype
        image = image.to(device=device, dtype=video_dtype)
        state = state.to(device=device, dtype=action_dtype)
        action = action.to(device=device, dtype=action_dtype)
        if image.min().item() < -1e-6 or image.max().item() > 1.0 + 1e-6:
            raise ValueError("Input images must be scaled to [0, 1]")
        with torch.no_grad():
            condition_latent = self.wam.compact_wan.encode_video(
                (image * 2.0 - 1.0).unsqueeze(2)
            )
            future_latent = self._make_future_noise(
                condition_latent, training=self.training
            )
            scheduler = self.scheduler_class(
                shift=self.flow_shift, sigma_min=0.0, extra_one_step=True
            )
            scheduler.set_timesteps(num_inference_steps=self.rollout_steps, training=False)
            timesteps = scheduler.timesteps.to(device=device, dtype=video_dtype)
            for timestep in timesteps[: self.rollout_steps]:
                current_t = timestep.expand(image.shape[0])
                velocity = self._joint_forward(
                    condition_latent,
                    future_latent,
                    state,
                    action,
                    text_embeddings,
                    current_t,
                    return_features=False,
                )["video_pred"]
                # The upstream scheduler resolves one scalar schedule index; the
                # expanded tensor is only for the batched Efficient-WAM forward.
                future_latent = scheduler.step(velocity, timestep, future_latent)

        final_t = torch.zeros(image.shape[0], device=device, dtype=video_dtype)
        grad_context = nullcontext() if self._has_trainable_wam_parameters else torch.no_grad()
        with grad_context:
            features = self._joint_forward(
                condition_latent,
                future_latent.detach(),
                state,
                action,
                text_embeddings,
                final_t,
                return_features=True,
            )
        return {
            "state_feature_taps": features["state_feature_taps"],
            "action_feature_taps": features["action_feature_taps"],
            "register_feature_taps": features["register_feature_taps"],
        }


def _checkpoint_model_config(payload: dict[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    config = payload.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise RuntimeError(
            f"Efficient-WAM checkpoint lacks exported model metadata: {checkpoint_path}"
        )
    return config["model"]


def build_model(config: dict[str, Any], device: str | torch.device | None = None) -> SafetyVerifyWAM:
    backbone_config = config["backbone"]
    if backbone_config.get("kind", "efficient_wam") != "efficient_wam":
        raise ValueError("Production builds only support backbone.kind=efficient_wam")
    source_root = resolve_project_path(backbone_config["source_root"])
    expected_commit = str(
        backbone_config.get("source_commit", EXPECTED_EFFICIENT_WAM_COMMIT)
    )
    _verify_external_source(source_root, expected_commit)
    modules = _import_external_modules(source_root)

    checkpoint_path = resolve_project_path(backbone_config["base_checkpoint"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Efficient-WAM checkpoint does not exist: {checkpoint_path}")
    payload = _trusted_torch_load(checkpoint_path)
    exported_model = _checkpoint_model_config(payload, checkpoint_path)
    compact_meta = exported_model["compact_wan"]
    action_meta = exported_model["action_expert"]
    future_video_size = compact_meta.get("future_video_size")
    if not future_video_size:
        raise RuntimeError("Base checkpoint has no multiscale future_video_size")

    selected_device = torch.device(device or config.get("device", "cuda"))
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    precision = str(backbone_config.get("precision", compact_meta.get("precision", "bfloat16")))
    wan_root = resolve_project_path(backbone_config["wan_root"])
    vae_path = resolve_project_path(
        backbone_config.get("vae_path", wan_root / "Wan2.2_VAE.pth")
    )
    compact_config = modules["CompactWANConfig"](
        checkpoint_path=str(wan_root),
        config_path=str(wan_root),
        vae_path=str(vae_path),
        precision=precision,
        dim=int(compact_meta["dim"]),
        ffn_dim=int(compact_meta["ffn_dim"]),
        num_heads=int(compact_meta["num_heads"]),
        num_layers=int(compact_meta["num_layers"]),
        head_dim=int(compact_meta.get("head_dim", 128)),
        future_video_size=tuple(int(value) for value in future_video_size),
    )
    LOGGER.info("Building compact Efficient-WAM architecture on %s", selected_device)
    compact_wam = modules["CompactWANModel"].from_config(
        compact_config, device=str(selected_device)
    )
    wam_config = modules["SmallWAMActionConfig"](
        compact_wan=compact_config,
        action_dim=int(action_meta["action_dim"]),
        state_dim=int(action_meta["state_dim"]),
        chunk_size=int(action_meta["chunk_size"]),
        ae_dim=int(action_meta["dim"]),
        ae_ffn_dim=int(action_meta["ffn_dim"]),
        ae_num_layers=int(action_meta["num_layers"]),
        wan_frozen=True,
    )
    wam = modules["SmallWAMActionModel"](wam_config, compact_wam)
    state = payload.get("model", payload)
    if not isinstance(state, dict):
        raise TypeError(f"Invalid Efficient-WAM model state in {checkpoint_path}")
    missing, unexpected = wam.load_state_dict(_strip_module_prefix(state), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Base Efficient-WAM checkpoint does not match its metadata: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    wam.to(selected_device)
    wam.eval()

    head_meta = config.get("model", {})
    num_taps = int(head_meta.get("num_taps", 2))
    adapter = EfficientWAMSafetyBackbone(
        wam,
        modules["FlowMatchScheduler"],
        num_video_frames=int(backbone_config.get("num_video_frames", 8)),
        rollout_steps=int(backbone_config.get("rollout_steps", 4)),
        flow_shift=float(backbone_config.get("flow_shift", 5.0)),
        rollout_seed=int(backbone_config.get("rollout_seed", 17)),
        randomize_training_noise=bool(
            backbone_config.get("randomize_training_noise", True)
        ),
        safety_tap_layers=num_taps,
    )
    adapter.configure_trainability(str(backbone_config.get("train_mode", "head_only")))
    head_config = RiskHeadConfig(
        action_dim=adapter.action_feature_dim,
        rank=int(head_meta.get("rank", 16)),
        alpha=float(head_meta.get("alpha", 16.0)),
        num_taps=num_taps,
        dropout=float(head_meta.get("dropout", 0.1)),
        max_action_steps=int(head_meta.get("max_action_steps", adapter.action_steps)),
        num_risk_types=int(head_meta.get("num_risk_types", 0)),
    )
    model = SafetyVerifyWAM(adapter, SafetyRiskHead(head_config)).to(selected_device)
    return model
