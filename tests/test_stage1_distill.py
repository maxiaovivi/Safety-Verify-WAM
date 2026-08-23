from __future__ import annotations

import unittest

import torch

from safety_verify_wam.stage1.aha_teacher import AHAOVCRSTeacherBatch
from safety_verify_wam.stage1.distill import Stage1LossConfig, stage1_distillation_loss


class Stage1DistillationLossTest(unittest.TestCase):
    def test_efficient_conditioning_skips_aha_structural_targets(self) -> None:
        prediction = torch.zeros(1, 2, 2)
        teacher_action = torch.ones_like(prediction)
        targets = AHAOVCRSTeacherBatch(
            noisy_action=teacher_action.clone(),
            action_t=torch.tensor([500.0]),
            sigma=torch.tensor([0.5]),
            teacher_velocity=torch.ones_like(prediction),
            teacher_action=teacher_action,
            ground_truth_action=teacher_action,
            initial_state=torch.zeros(1, 2),
            reference_velocity=torch.full_like(prediction, 2.0),
            observation_tokens=torch.empty(1, 0, 2),
            observation_mask=torch.empty(1, 0, dtype=torch.bool),
            video_kv_cache=tuple(),
            teacher_queries=torch.empty(1, 0, 2),
            teacher_editor_trace={},
            teacher_action_responses={},
            action_is_pad=None,
            chunk_index=torch.zeros(1, dtype=torch.long),
            anchor_step=torch.zeros(1, dtype=torch.long),
        )
        loss, terms = stage1_distillation_loss(
            {"action_velocity": prediction},
            targets,
            Stage1LossConfig(
                velocity_weight=1.0,
                teacher_action_weight=1.0,
                ground_truth_action_weight=0.25,
                preservation_weight=0.25,
                query_weight=0.0,
                route_weight=0.0,
                delta_weight=0.0,
                response_weight=0.0,
            ),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(terms["loss_query"]), 0.0)
        self.assertEqual(float(terms["loss_route"]), 0.0)
        self.assertEqual(float(terms["loss_delta"]), 0.0)
        self.assertEqual(float(terms["loss_response"]), 0.0)
        self.assertGreater(float(terms["loss_preservation"]), 0.0)


if __name__ == "__main__":
    unittest.main()
