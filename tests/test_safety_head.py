from __future__ import annotations

import unittest

import torch

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
                video_dim=8,
                action_dim=6,
                hidden_dim=12,
                num_heads=3,
                num_layers=1,
                dropout=0.0,
                max_action_steps=4,
            )
        ).eval()

    def test_action_queries_produce_three_class_chunk_and_step_logits(self) -> None:
        head = self._head()
        outputs = head(
            condition_features=torch.randn(2, 5, 8),
            future_features=torch.randn(2, 3, 8),
            state_features=torch.randn(2, 1, 6),
            action_features=torch.randn(2, 4, 6),
        )

        self.assertEqual(tuple(outputs["class_logits"].shape), (2, 3))
        self.assertEqual(tuple(outputs["step_class_logits"].shape), (2, 4, 3))

    def test_safety_model_exposes_probabilities_and_ordinal_severity(self) -> None:
        class FakeBackbone(torch.nn.Module):
            def imagine(self, image, state, action, text_embeddings):
                batch = image.shape[0]
                self.seen_state = state
                self.seen_text = text_embeddings
                return {
                    "condition_features": torch.randn(batch, 5, 8),
                    "future_features": torch.randn(batch, 3, 8),
                    "state_features": torch.randn(batch, 1, 6),
                    "action_features": torch.randn(batch, action.shape[1], 6),
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
