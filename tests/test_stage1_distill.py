from __future__ import annotations

import unittest
from dataclasses import replace
from types import MethodType

import torch

from safety_verify_wam.stage1.aha_teacher import (
    AHAOVCRSTeacherBatch,
    AHAOVCRTeacherAdapter,
    GroundTruthTargetAdapter,
)
from safety_verify_wam.stage1.distill import (
    AHAOVCRSStage1Program,
    Stage1LossConfig,
    _resolve_parameter_dtype,
    stage1_distillation_loss,
)
from safety_verify_wam.stage1.ovcr_s import OVCRSActionGenerator, OVCRSConfig


class Stage1DistillationLossTest(unittest.TestCase):
    @staticmethod
    def _tiny_config() -> OVCRSConfig:
        return OVCRSConfig(
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

    def test_ground_truth_adapter_builds_chunks_without_teacher(self) -> None:
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
        adapter = GroundTruthTargetAdapter(
            config,
            action_horizon=4,
            device="cpu",
        )
        action = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2)
        proprio = action + 100.0
        sample = {
            "action": action,
            "proprio": proprio,
            "action_is_pad": torch.tensor(
                [[False, False, True, True], [False, False, False, False]]
            ),
            "stage1_chunk_index": torch.tensor([0, 1]),
        }

        targets = adapter.prepare_batch(sample)

        torch.testing.assert_close(targets.ground_truth_action[0], action[0, :2])
        torch.testing.assert_close(targets.ground_truth_action[1], action[1, 2:])
        torch.testing.assert_close(targets.teacher_action, targets.ground_truth_action)
        torch.testing.assert_close(targets.initial_state[0], proprio[0, 0])
        torch.testing.assert_close(targets.initial_state[1], proprio[1, 2])
        self.assertEqual(targets.action_is_pad.tolist(), [[False, False], [False, False]])
        self.assertEqual(adapter.teacher_layer_mapping, ())
        self.assertEqual(sum(parameter.numel() for parameter in adapter.parameters()), 0)

    def test_ground_truth_adapter_rejects_action_offset(self) -> None:
        adapter = GroundTruthTargetAdapter(
            OVCRSConfig(), action_horizon=64, device="cpu"
        )
        with self.assertRaisesRegex(ValueError, "action_offset"):
            adapter.prepare_batch(
                {
                    "action": torch.zeros(1, 64, 14),
                    "proprio": torch.zeros(1, 64, 14),
                    "action_offset": torch.zeros(1, dtype=torch.long),
                }
            )

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

    def test_response_loss_projects_both_models_to_requested_dimension(self) -> None:
        prediction = torch.zeros(1, 1, 1)
        targets = AHAOVCRSTeacherBatch(
            noisy_action=prediction.clone(),
            action_t=torch.zeros(1),
            sigma=torch.zeros(1),
            teacher_velocity=prediction.clone(),
            teacher_action=prediction.clone(),
            ground_truth_action=prediction.clone(),
            initial_state=torch.zeros(1, 1),
            reference_velocity=None,
            observation_tokens=torch.empty(1, 0, 1),
            observation_mask=torch.empty(1, 0, dtype=torch.bool),
            video_kv_cache=tuple(),
            teacher_queries=torch.empty(1, 0, 1),
            teacher_editor_trace={},
            teacher_action_responses={
                1: torch.tensor([[[1.0, 1.0, 0.0, 0.0]]])
            },
            action_is_pad=None,
            chunk_index=torch.zeros(1, dtype=torch.long),
            anchor_step=torch.zeros(1, dtype=torch.long),
        )
        config = Stage1LossConfig(
            velocity_weight=0.0,
            teacher_action_weight=0.0,
            ground_truth_action_weight=0.0,
            query_weight=0.0,
            route_weight=0.0,
            delta_weight=0.0,
            response_weight=1.0,
            response_projection_dim=2,
        )
        loss, terms = stage1_distillation_loss(
            {
                "action_velocity": prediction,
                "action_responses": {
                    1: torch.tensor([[[1.0, 0.0, 0.0, 1.0]]])
                },
            },
            targets,
            config,
        )
        self.assertAlmostEqual(float(loss), 1.0 - 2.0**-0.5, places=6)
        self.assertAlmostEqual(float(terms["loss_response"]), float(loss), places=6)
        with self.assertRaisesRegex(ValueError, "response_projection_dim"):
            replace(config, response_projection_dim=0)

    def test_response_targets_use_efficient_sampled_action(self) -> None:
        config = self._tiny_config()
        student = OVCRSActionGenerator(config)

        class DummyTeacherAdapter(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.student_config = config
                self.teacher_layer_mapping = (1,)
                self.converted_action = None

            def prepare_batch(self, _sample, *, tiled=False):
                del tiled
                action = torch.ones(1, 2, 2)
                return AHAOVCRSTeacherBatch(
                    noisy_action=action,
                    action_t=torch.tensor([500.0]),
                    sigma=torch.tensor([0.5]),
                    teacher_velocity=torch.zeros_like(action),
                    teacher_action=action,
                    ground_truth_action=action,
                    initial_state=torch.zeros(1, 2),
                    reference_velocity=None,
                    observation_tokens=torch.ones(1, 1, 2),
                    observation_mask=torch.ones(1, 1, dtype=torch.bool),
                    video_kv_cache=(
                        {"k": torch.ones(1, 1, 4), "v": torch.ones(1, 1, 4)},
                    ),
                    teacher_queries=torch.empty(1, 0, 4),
                    teacher_editor_trace={},
                    teacher_action_responses={},
                    action_is_pad=None,
                    chunk_index=torch.zeros(1, dtype=torch.long),
                    anchor_step=torch.zeros(1, dtype=torch.long),
                    response_context={"video_state": {}},
                )

            def attach_action_response_targets(self, targets, noisy_action):
                self.converted_action = noisy_action.detach().clone()
                return replace(
                    targets,
                    teacher_action_responses={1: torch.ones(1, 2, 4)},
                    response_context=None,
                )

        class DummyEfficientAdapter(torch.nn.Module):
            freeze_action_expert = True

            def prepare_batch(self, _sample, targets):
                return replace(targets, noisy_action=torch.full((1, 2, 2), 3.0))

            def action_efficient_to_aha(self, action):
                return action * 2.0

        teacher_adapter = DummyTeacherAdapter()
        program = AHAOVCRSStage1Program(
            teacher_adapter,
            student,
            Stage1LossConfig(
                velocity_weight=1.0,
                teacher_action_weight=0.0,
                ground_truth_action_weight=0.0,
                query_weight=0.0,
                route_weight=0.0,
                delta_weight=0.0,
                response_weight=0.2,
                response_projection_dim=2,
            ),
            efficient_training_adapter=DummyEfficientAdapter(),
        )
        loss, terms = program.training_loss({})
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("loss_response", terms)
        torch.testing.assert_close(
            teacher_adapter.converted_action, torch.full((1, 2, 2), 6.0)
        )

    def test_response_adapter_replaces_only_the_selected_action_chunk(self) -> None:
        config = self._tiny_config()
        adapter = AHAOVCRTeacherAdapter.__new__(AHAOVCRTeacherAdapter)
        torch.nn.Module.__init__(adapter)
        adapter.student_config = config
        observed = {}

        def fake_predict(_self, *, noisy_action, timestep_action, video_state):
            observed["action"] = noisy_action.detach().clone()
            observed["timestep"] = timestep_action.detach().clone()
            self.assertIs(video_state, context_video_state)
            response = torch.cat([noisy_action, noisy_action], dim=-1)
            return torch.zeros_like(noisy_action), {1: response}

        adapter._predict_velocity_with_response_trace = MethodType(fake_predict, adapter)
        context_video_state = {"action": torch.zeros(1, 4, 2)}
        targets = AHAOVCRSTeacherBatch(
            noisy_action=torch.ones(1, 2, 2),
            action_t=torch.tensor([250.0]),
            sigma=torch.tensor([0.25]),
            teacher_velocity=torch.zeros(1, 2, 2),
            teacher_action=torch.zeros(1, 2, 2),
            ground_truth_action=torch.zeros(1, 2, 2),
            initial_state=torch.zeros(1, 2),
            reference_velocity=None,
            observation_tokens=torch.empty(1, 0, 2),
            observation_mask=torch.empty(1, 0, dtype=torch.bool),
            video_kv_cache=tuple(),
            teacher_queries=torch.empty(1, 0, 4),
            teacher_editor_trace={},
            teacher_action_responses={},
            action_is_pad=None,
            chunk_index=torch.ones(1, dtype=torch.long),
            anchor_step=torch.zeros(1, dtype=torch.long),
            response_context={"video_state": context_video_state},
        )

        attached = adapter.attach_action_response_targets(
            targets, torch.ones(1, 2, 2)
        )

        torch.testing.assert_close(observed["action"][:, :2], torch.zeros(1, 2, 2))
        torch.testing.assert_close(observed["action"][:, 2:], torch.ones(1, 2, 2))
        torch.testing.assert_close(
            observed["timestep"], torch.full((1, 2), 250.0)
        )
        torch.testing.assert_close(
            attached.teacher_action_responses[1], torch.ones(1, 2, 4)
        )
        self.assertIsNone(attached.response_context)


if __name__ == "__main__":
    unittest.main()
