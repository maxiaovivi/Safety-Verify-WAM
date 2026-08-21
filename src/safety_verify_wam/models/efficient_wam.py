from __future__ import annotations

import importlib
import logging
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

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
    """Pinned Efficient-WAM adapter for action-conditioned future imagination.

    The external model remains unmodified. This adapter reuses its action MLP,
    joint attention blocks, video head, and VAE while removing sample-level
    robot state and task text from the safety network interface.
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
    ) -> None:
        super().__init__()
        self.wam = wam
        self.scheduler_class = scheduler_class
        self.num_video_frames = int(num_video_frames)
        self.rollout_steps = int(rollout_steps)
        self.flow_shift = float(flow_shift)
        self.rollout_seed = int(rollout_seed)
        self.randomize_training_noise = bool(randomize_training_noise)
        self.video_dim = int(wam.config.compact_wan.dim)
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
        self.wam.configure_teacache(enabled=False)

    def train(self, mode: bool = True) -> "EfficientWAMSafetyBackbone":
        super().train(mode)
        # Efficient-WAM has no safety-specific running statistics. Keeping it in
        # eval mode makes rollout repeatable while still allowing parameter grads.
        self.wam.eval()
        return self

    def configure_trainability(self, mode: str, last_joint_layers: int = 2) -> None:
        for parameter in self.wam.parameters():
            parameter.requires_grad_(False)
        mode = str(mode)
        count = max(1, int(last_joint_layers))
        if mode == "head_only":
            pass
        elif mode == "action_last":
            encoder = self.wam.action_expert.input_encoder.action_encoder
            for parameter in encoder.parameters():
                parameter.requires_grad_(True)
            for block in self.wam.action_expert.blocks[-count:]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
        elif mode == "joint_last":
            encoder = self.wam.action_expert.input_encoder.action_encoder
            for parameter in encoder.parameters():
                parameter.requires_grad_(True)
            for block in self.wam.action_expert.blocks[-count:]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
            video_blocks = self.wam.compact_wan.video_model.wan_model.blocks
            for block in video_blocks[-count:]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
        elif mode == "all_except_vae":
            for parameter in self.wam.parameters():
                parameter.requires_grad_(True)
        else:
            raise ValueError(
                f"Unsupported backbone.train_mode={mode!r}; expected head_only, "
                "action_last, joint_last, or all_except_vae"
            )
        vae = getattr(self.wam.compact_wan.video_model, "vae", None)
        if vae is not None and hasattr(vae, "parameters"):
            for parameter in vae.parameters():
                parameter.requires_grad_(False)
        self._has_trainable_wam_parameters = any(
            parameter.requires_grad for parameter in self.wam.parameters()
        )
        self.wam.eval()

    def _encode_actions_only(
        self,
        action: torch.Tensor,
        registers: torch.Tensor | None,
    ) -> torch.Tensor:
        encoder = self.wam.action_expert.input_encoder
        if not hasattr(encoder, "action_encoder") or not hasattr(encoder, "pos_embedding"):
            raise RuntimeError("Pinned Efficient-WAM action encoder interface changed")
        encoded_action = encoder.action_encoder(action)
        action_steps = int(encoded_action.shape[1])
        action_positions = encoder.pos_embedding[:, 1 : 1 + action_steps]
        action_positions = action_positions.to(
            device=encoded_action.device, dtype=encoded_action.dtype
        )
        encoded_action = encoded_action + action_positions
        if registers is None:
            return encoded_action
        register_steps = int(registers.shape[1])
        register_start = 1 + action_steps
        register_positions = encoder.pos_embedding[
            :, register_start : register_start + register_steps
        ].to(device=registers.device, dtype=registers.dtype)
        return torch.cat([encoded_action, registers + register_positions], dim=1)

    def _null_text_embeddings(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        text_dim = int(self.wam.compact_wan.video_model.wan_model.text_dim)
        return [torch.zeros((1, text_dim), device=device, dtype=dtype) for _ in range(batch_size)]

    def _joint_forward(
        self,
        condition_latent: torch.Tensor,
        future_latent: torch.Tensor,
        action: torch.Tensor,
        video_t: torch.Tensor,
        *,
        return_features: bool,
    ) -> dict[str, torch.Tensor]:
        wam = self.wam
        video_tokens, seq_lens, layout, freqs = wam.compact_wan.prepare_multiscale_video_tokens(
            condition_latent, future_latent
        )
        text_embeddings = self._null_text_embeddings(
            action.shape[0], video_tokens.device, video_tokens.dtype
        )
        text_context = wam.compact_wan.prepare_text_context(text_embeddings)
        video_head_time_emb, video_adaln_params = wam._build_video_time_embeddings(
            video_t, video_tokens.shape[1]
        )

        registers = None
        if wam.action_expert.config.num_registers > 0 and wam.action_expert.registers is not None:
            registers = wam.action_expert.registers.expand(action.shape[0], -1, -1)
        action_tokens = self._encode_actions_only(action, registers)
        action_t = torch.zeros(
            action.shape[0], device=action.device, dtype=action.dtype
        )
        _, action_adaln_params = wam._build_action_time_embeddings(
            action_t, action_tokens.shape[1]
        )

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

        outputs: dict[str, torch.Tensor] = {
            "video_pred": wam.compact_wan.apply_multiscale_video_head(
                video_tokens, video_head_time_emb, layout
            )
        }
        if return_features:
            condition_len = int(layout["condition_seq_len"])
            future_shape = tuple(int(value) for value in layout["future_grid_shape"])
            future_steps, future_h, future_w = future_shape
            future_tokens = video_tokens[:, condition_len:]
            expected_tokens = future_steps * future_h * future_w
            if future_tokens.shape[1] != expected_tokens:
                raise RuntimeError(
                    f"Future token layout mismatch: {future_tokens.shape[1]} != {expected_tokens}"
                )
            future_features = future_tokens.reshape(
                future_tokens.shape[0], future_steps, future_h * future_w, future_tokens.shape[-1]
            ).mean(dim=2)
            outputs["future_features"] = future_features
            outputs["action_features"] = action_tokens[:, : action.shape[1]]
        return outputs

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

    def imagine(self, image: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        if action.shape[1:] != (self.action_steps, self.action_dim):
            raise ValueError(
                f"Expected action [B, {self.action_steps}, {self.action_dim}], "
                f"got {tuple(action.shape)}"
            )
        model_parameter = next(self.wam.parameters())
        device = model_parameter.device
        video_dtype = self.wam.compact_wan.video_model.precision
        action_dtype = next(self.wam.action_expert.parameters()).dtype
        image = image.to(device=device, dtype=video_dtype)
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
                    action,
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
                action,
                final_t,
                return_features=True,
            )
        return {
            "future_features": features["future_features"],
            "action_features": features["action_features"],
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
    )
    adapter.configure_trainability(
        str(backbone_config.get("train_mode", "head_only")),
        int(backbone_config.get("train_last_joint_layers", 2)),
    )
    head_meta = config.get("model", {})
    head_config = RiskHeadConfig(
        video_dim=adapter.video_dim,
        action_dim=adapter.action_feature_dim,
        hidden_dim=int(head_meta.get("hidden_dim", 512)),
        num_heads=int(head_meta.get("num_heads", 8)),
        num_layers=int(head_meta.get("num_layers", 2)),
        dropout=float(head_meta.get("dropout", 0.1)),
        max_future_steps=int(head_meta.get("max_future_steps", 8)),
        max_action_steps=int(head_meta.get("max_action_steps", adapter.action_steps)),
        num_risk_types=int(head_meta.get("num_risk_types", 0)),
        unsafe_threshold=float(head_meta.get("unsafe_threshold", 0.5)),
    )
    model = SafetyVerifyWAM(adapter, SafetyRiskHead(head_config)).to(selected_device)
    return model
