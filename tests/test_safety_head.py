from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from safety_verify_wam.models.efficient_wam import EfficientWAMSafetyBackbone
from safety_verify_wam.models.safety_verifier import (
    RiskHeadConfig,
    SafetyRiskHead,
    SafetyVerifyWAM,
)
from safety_verify_wam.training.losses import safety_loss


class SafetyRiskHeadTest(unittest.TestCase):
    @staticmethod
    def _head() -> SafetyRiskHead:
        return SafetyRiskHead(
            RiskHeadConfig(
                action_dim=6,
                rank=2,
                alpha=2.0,
                num_taps=2,
                dropout=0.0,
                max_action_steps=4,
            )
        ).eval()

    def test_low_rank_taps_produce_three_class_chunk_and_step_logits(self) -> None:
        head = self._head()
        outputs = head(
            state_feature_taps=torch.randn(2, 2, 1, 6),
            action_feature_taps=torch.randn(2, 2, 4, 6),
            register_feature_taps=torch.randn(2, 2, 2, 6),
        )

        self.assertEqual(tuple(outputs["class_logits"].shape), (2, 3))
        self.assertEqual(tuple(outputs["step_class_logits"].shape), (2, 4, 3))
        self.assertEqual(tuple(outputs["risk_features"].shape), (2, 6))
        self.assertEqual(tuple(outputs["step_risk_features"].shape), (2, 4, 6))
        torch.testing.assert_close(
            outputs["safety_pool_weights"].sum(dim=-1),
            torch.ones(2),
        )
        torch.testing.assert_close(
            outputs["safety_tap_weights"].sum(),
            torch.tensor(1.0),
        )

    def test_low_rank_head_is_small_and_zero_initialized(self) -> None:
        head = SafetyRiskHead(
            RiskHeadConfig(action_dim=768, rank=16, num_taps=2)
        )
        trainable = sum(parameter.numel() for parameter in head.parameters())

        self.assertLess(trainable, 100_000)
        for adapter in head.tap_adapters:
            torch.testing.assert_close(
                adapter.up.weight,
                torch.zeros_like(adapter.up.weight),
            )
            tokens = torch.randn(2, 5, 768)
            torch.testing.assert_close(adapter(tokens), tokens)

    def test_backbone_mode_freezes_every_efficient_parameter(self) -> None:
        class FakeCompact(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(2, 2))
                self.is_multiscale = True
                self.video_model = SimpleNamespace(vae=None)

        class FakeWAM(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.compact_wan = FakeCompact()
                self.action_weight = torch.nn.Parameter(torch.randn(2, 2))
                self.config = SimpleNamespace(
                    compact_wan=SimpleNamespace(dim=8),
                    ae_dim=6,
                    action_dim=6,
                    chunk_size=4,
                    ae_num_layers=2,
                )

            def configure_teacache(self, enabled: bool) -> None:
                self.teacache_enabled = enabled

        backbone = EfficientWAMSafetyBackbone(
            FakeWAM(),
            object,
            num_video_frames=8,
            rollout_steps=2,
            flow_shift=5.0,
            rollout_seed=17,
            randomize_training_noise=False,
            safety_tap_layers=2,
        )
        backbone.configure_trainability("head_only")

        self.assertFalse(
            any(parameter.requires_grad for parameter in backbone.parameters())
        )
        with self.assertRaisesRegex(ValueError, "must be 'head_only'"):
            backbone.configure_trainability("action_last")

    def test_safety_model_exposes_probabilities_and_ordinal_severity(self) -> None:
        class FakeBackbone(torch.nn.Module):
            def imagine(self, image, state, action, text_embeddings):
                batch = image.shape[0]
                self.seen_state = state
                self.seen_text = text_embeddings
                return {
                    "state_feature_taps": torch.randn(batch, 2, 1, 6),
                    "action_feature_taps": torch.randn(
                        batch, 2, action.shape[1], 6
                    ),
                    "register_feature_taps": torch.randn(batch, 2, 2, 6),
                }

        backbone = FakeBackbone()
        model = SafetyVerifyWAM(backbone, self._head()).eval()
        state = torch.randn(2, 6)
        text_embeddings = torch.randn(2, 7, 9)
        outputs = model.predict(
            torch.randn(2, 3, 16, 16),
            state,
            torch.randn(2, 4, 6),
            text_embeddings,
        )

        self.assertEqual(tuple(outputs["class_probabilities"].shape), (2, 3))
        torch.testing.assert_close(
            outputs["class_probabilities"].sum(dim=-1),
            torch.ones(2),
        )
        self.assertTrue(torch.all(outputs["severity_score"] >= 0))
        self.assertTrue(torch.all(outputs["severity_score"] <= 1))
        self.assertEqual(tuple(outputs["requires_intervention"].shape), (2,))
        torch.testing.assert_close(backbone.seen_state, state)
        torch.testing.assert_close(backbone.seen_text, text_embeddings)

    def test_loss_uses_three_class_cross_entropy(self) -> None:
        outputs = {
            "class_logits": torch.tensor(
                [[5.0, 0.0, -1.0], [-1.0, 4.0, 0.0], [0.0, -1.0, 5.0]],
                requires_grad=True,
            ),
            "step_class_logits": torch.randn(3, 2, 3, requires_grad=True),
        }
        batch = {
            "risk": torch.tensor([0, 1, 2]),
            "risk_steps": torch.full((3, 2), -100, dtype=torch.long),
            "risk_type": torch.full((3,), -100, dtype=torch.long),
        }

        loss, terms = safety_loss(outputs, batch, {"chunk_weight": 1.0}, "cpu")

        self.assertLess(float(loss.item()), 0.05)
        self.assertIn("chunk_class_loss", terms)
        loss.backward()
        self.assertIsNotNone(outputs["class_logits"].grad)


if __name__ == "__main__":
    unittest.main()
