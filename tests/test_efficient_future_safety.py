from __future__ import annotations

import torch

from safety_verify_wam.portable.contracts import SafetyBatch
from safety_verify_wam.portable.future import (
    EfficientFutureSafetyConfig,
    EfficientFutureSafetySidecar,
    extract_future_value_tokens,
)
from safety_verify_wam.portable.multidomain import (
    MultiProfilePortableSafetyCore,
    MultiProfileSafetyConfig,
    ProfileAdapterConfig,
)


def _batch(batch_size: int = 2) -> SafetyBatch:
    return SafetyBatch(
        video=torch.rand(batch_size, 1, 3, 3, 32, 32),
        state=torch.rand(batch_size, 1, 14),
        action=torch.rand(batch_size, 16, 14),
        video_timestamps=torch.zeros(batch_size, 1),
        state_timestamps=torch.zeros(batch_size, 1),
        action_timestamps=torch.arange(1, 17).float().repeat(batch_size, 1) / 10,
        video_mask=torch.ones(batch_size, 1, 3, dtype=torch.bool),
        state_mask=torch.ones(batch_size, 1, dtype=torch.bool),
        action_mask=torch.ones(batch_size, 16, dtype=torch.bool),
    )


def _model() -> EfficientFutureSafetySidecar:
    base = MultiProfilePortableSafetyCore(
        MultiProfileSafetyConfig(
            profiles=(
                ProfileAdapterConfig(
                    key="bimanual_qpos14",
                    state_dim=14,
                    action_dim=14,
                    motion_mode="position_target",
                ),
            ),
            model_dim=32,
            vision_channels=(8, 16),
            transformer_layers=1,
            attention_heads=4,
            max_views=3,
            profile_specific_heads=True,
        )
    )
    return EfficientFutureSafetySidecar(
        base,
        EfficientFutureSafetyConfig(future_dim=64, attention_heads=4),
    ).eval()


def test_extract_future_tokens_preserves_every_position() -> None:
    values = torch.arange(2 * 7 * 2 * 3).reshape(2, 7, 2, 3).float()
    cache = {
        "video_v": [torch.zeros_like(values), values],
        "grid_sizes": {"condition_seq_len": 3, "future_seq_len": 4},
    }
    result = extract_future_value_tokens(cache)
    assert result.shape == (2, 4, 6)
    assert torch.equal(result, values[:, 3:7].flatten(2))
    assert result.requires_grad is False


def test_zero_initialized_future_path_preserves_checkpoint_logits() -> None:
    torch.manual_seed(3)
    model = _model()
    batch = _batch()
    future = torch.randn(2, 20, 64)
    without_future = model("bimanual_qpos14", batch, future_mode="none")
    with_future = model(
        "bimanual_qpos14", batch, future, future_mode="full"
    )
    assert torch.equal(without_future["class_logits"], with_future["class_logits"])
    assert torch.equal(
        without_future["step_class_logits"], with_future["step_class_logits"]
    )


def test_future_adapter_trains_without_touching_base() -> None:
    torch.manual_seed(5)
    model = _model().train()
    batch = _batch()
    future = torch.randn(2, 20, 64, requires_grad=True)
    output = model("bimanual_qpos14", batch, future, future_mode="full")
    output["class_logits"].sum().backward()
    assert all(parameter.grad is None for parameter in model.base.parameters())
    assert model.future_attention.out_proj.weight.grad is not None
    assert future.grad is None


def test_mean_mode_reduces_only_the_future_sequence() -> None:
    model = _model()
    batch = _batch()
    future = torch.randn(2, 20, 64)
    output = model("bimanual_qpos14", batch, future, future_mode="mean")
    assert output["future_token_count"].item() == 1
    assert output["step_class_logits"].shape == (2, 16, 2)
