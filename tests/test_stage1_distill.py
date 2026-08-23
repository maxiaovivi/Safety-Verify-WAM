from __future__ import annotations

import unittest

import torch

from safety_verify_wam.stage1.aha_teacher import AHAOVCRSTeacherBatch
from safety_verify_wam.stage1.distill import (
    AHAOVCRSStage1Program,
    Stage1LossConfig,
    _resolve_parameter_dtype,
    stage1_distillation_loss,
)
from safety_verify_wam.stage1.ovcr_s import OVCRSActionGenerator, OVCRSConfig


class Stage1DistillationLossTest(unittest.TestCase):
    def test_editor_only_view_survives_aha_trainer_freeze_sequence(self) -> None:
        config = OVCRSConfig(
            observation_dim=2,
            query_dim=4,
            num_queries=2,
            video_dim=4,
            num_heads=1,
            head_dim=4,
            num_layers=1,
            editor_rank=2,
            state_dim=2,
            action_dim=2,
            action_hidden_dim=4,
            action_ffn_dim=8,
            action_chunk_size=2,
            num_registers=1,
            time_embedding_dim=4,
            distill_layers=(1,),
        )
        student = OVCRSActionGenerator(config)
        teacher_adapter = torch.nn.Module()
        teacher_adapter.student_config = config
        efficient_adapter = torch.nn.Module()
        efficient_adapter.freeze_action_expert = True
        program = AHAOVCRSStage1Program(
            teacher_adapter,
            student,
            Stage1LossConfig(),
            efficient_training_adapter=efficient_adapter,
        )

        program.eval()
        program.requires_grad_(False)
        program.dit.train()
        program.dit.requires_grad_(True)

        self.assertTrue(
            all(parameter.requires_grad for parameter in student.query_encoder.parameters())
        )
        self.assertTrue(
            all(parameter.requires_grad for parameter in student.cache_editor.parameters())
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in student.action_expert.parameters()
            )
        )

    def test_student_parameter_dtype_supports_fp32_master_weights(self) -> None:
        self.assertIs(
            _resolve_parameter_dtype("float32", fallback=torch.bfloat16),
            torch.float32,
        )
        self.assertIs(
            _resolve_parameter_dtype(None, fallback=torch.bfloat16),
            torch.bfloat16,
        )
        with self.assertRaisesRegex(ValueError, "student_parameter_dtype"):
            _resolve_parameter_dtype("float16", fallback=torch.bfloat16)

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
