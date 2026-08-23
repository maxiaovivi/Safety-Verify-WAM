from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from safety_verify_wam.stage1.efficient_adapter import (
    AHAActionDenormalizer,
    AHAEfficientNormalizerBridge,
    compact_first_frame_video_cache,
    compact_video_cache,
    load_ovcrs_student,
    observation_tokens_from_condition_latent,
    share_efficient_action_expert,
)
from safety_verify_wam.stage1.efficient_training import _shared_batch_timestep
from safety_verify_wam.stage1.ovcr_s import OVCRSActionGenerator, OVCRSConfig


class EfficientAdapterTest(unittest.TestCase):
    def test_compact_action_expert_backpropagates_to_encoder_and_ffn(self) -> None:
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
        torch.nn.init.normal_(student.action_expert.decoder.action_head[0].weight)
        outputs = student(
            noisy_action=torch.randn(2, 2, 2),
            action_t=torch.tensor([250.0, 750.0]),
            initial_state=torch.randn(2, 2),
            observation_tokens=torch.randn(2, 3, 2),
            video_kv_cache=(
                {"k": torch.randn(2, 3, 4), "v": torch.randn(2, 3, 4)},
            ),
        )

        outputs["action_velocity"].square().mean().backward()

        encoder_grads = [
            parameter.grad
            for parameter in student.action_expert.input_encoder.parameters()
        ]
        ffn_grads = [
            parameter.grad
            for parameter in student.action_expert.blocks[0].ffn.parameters()
        ]
        self.assertTrue(any(grad is not None and grad.abs().sum() > 0 for grad in encoder_grads))
        self.assertTrue(any(grad is not None and grad.abs().sum() > 0 for grad in ffn_grads))

    def test_efficient_runtime_and_student_share_one_action_expert(self) -> None:
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
        runtime_model = torch.nn.Module()
        runtime_model.action_expert = OVCRSActionGenerator(config).action_expert
        runtime_model.action_expert.freq_dim = config.time_embedding_dim

        shared = share_efficient_action_expert(runtime_model, student)

        self.assertIs(shared, student.action_expert)
        self.assertIs(runtime_model.action_expert, student.action_expert)
        self.assertEqual(runtime_model.action_expert.freq_dim, config.time_embedding_dim)

    def test_action_expert_sharing_rejects_incompatible_shapes(self) -> None:
        student = OVCRSActionGenerator(
            OVCRSConfig(
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
        )
        incompatible = OVCRSActionGenerator(
            OVCRSConfig(
                observation_dim=2,
                query_dim=4,
                num_queries=2,
                video_dim=4,
                num_heads=1,
                head_dim=4,
                num_layers=1,
                editor_rank=2,
                state_dim=2,
                action_dim=3,
                action_hidden_dim=4,
                action_ffn_dim=8,
                action_chunk_size=2,
                num_registers=1,
                time_embedding_dim=4,
                distill_layers=(1,),
            )
        )
        runtime_model = torch.nn.Module()
        runtime_model.action_expert = incompatible.action_expert

        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            share_efficient_action_expert(runtime_model, student)

    def test_video_scheduler_receives_one_shared_batch_timestep(self) -> None:
        timestep = torch.tensor([750.0, 750.0, 750.0, 750.0])
        shared = _shared_batch_timestep(timestep)
        self.assertEqual(shared.ndim, 0)
        self.assertEqual(float(shared.item()), 750.0)
        with self.assertRaisesRegex(ValueError, "shared batch timestep"):
            _shared_batch_timestep(torch.tensor([750.0, 250.0]))

    def test_observation_tokens_preserve_spatial_order(self) -> None:
        latent = torch.arange(8, dtype=torch.float32).reshape(1, 2, 1, 2, 2)
        tokens = observation_tokens_from_condition_latent(latent)
        self.assertEqual(tuple(tokens.shape), (1, 4, 2))
        torch.testing.assert_close(tokens[0, 0], torch.tensor([0.0, 4.0]))
        torch.testing.assert_close(tokens[0, 3], torch.tensor([3.0, 7.0]))

    def test_multiscale_cache_keeps_only_condition_tokens(self) -> None:
        keys = [torch.arange(48, dtype=torch.float32).reshape(1, 6, 2, 4) for _ in range(2)]
        values = [tensor + 100.0 for tensor in keys]
        compact = compact_first_frame_video_cache(
            {
                "grid_sizes": {"condition_seq_len": 3, "future_seq_len": 3},
                "video_k": keys,
                "video_v": values,
            },
            expected_layers=2,
            expected_dim=8,
        )
        self.assertEqual(len(compact), 2)
        self.assertEqual(tuple(compact[0]["k"].shape), (1, 3, 8))
        torch.testing.assert_close(compact[0]["v"], compact[0]["k"] + 100.0)

    def test_single_scale_cache_derives_first_frame_size(self) -> None:
        keys = [torch.zeros(1, 12, 1, 4)]
        compact = compact_first_frame_video_cache(
            {
                "grid_sizes": torch.tensor([[3, 2, 2]]),
                "video_k": keys,
                "video_v": keys,
            },
            expected_layers=1,
            expected_dim=4,
        )
        self.assertEqual(tuple(compact[0]["k"].shape), (1, 4, 4))

    def test_full_cache_preserves_condition_and_future_tokens(self) -> None:
        keys = [torch.arange(48, dtype=torch.float32).reshape(1, 6, 2, 4)]
        compact = compact_video_cache(
            {
                "grid_sizes": {"condition_seq_len": 3, "future_seq_len": 3},
                "video_k": keys,
                "video_v": keys,
            },
            expected_layers=1,
            expected_dim=8,
        )
        self.assertEqual(tuple(compact[0]["k"].shape), (1, 6, 8))

    def test_aha_action_denormalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.json"
            path.write_text(
                json.dumps(
                    {
                        "action": {
                            "default": {
                                "global_mean": [1.0, -2.0],
                                "global_std": [0.5, 4.0],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            normalizer = AHAActionDenormalizer.from_dataset_stats(path)
        action = normalizer.denormalize(torch.tensor([[[2.0, 0.5]]]))
        torch.testing.assert_close(action, torch.tensor([[[2.0, 0.0]]]))

    def test_aha_to_efficient_normalization_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aha_path = root / "aha.json"
            efficient_path = root / "efficient.json"
            aha_path.write_text(
                json.dumps(
                    {
                        "action": {
                            "default": {
                                "global_mean": [1.0, -2.0],
                                "global_std": [0.5, 4.0],
                            }
                        },
                        "state": {
                            "default": {
                                "global_mean": [3.0, 1.0],
                                "global_std": [2.0, 0.25],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            efficient_path.write_text(
                json.dumps(
                    {
                        "robotwin_qpos": {
                            "mean": [2.0, 0.0],
                            "std": [2.0, 2.0],
                        }
                    }
                ),
                encoding="utf-8",
            )
            bridge = AHAEfficientNormalizerBridge.from_dataset_stats(
                aha_path, efficient_path
            )
        action = torch.tensor([[[2.0, 0.5]]])
        efficient_action = bridge.action_aha_to_efficient(action)
        torch.testing.assert_close(
            efficient_action, torch.tensor([[[0.0, 0.0]]])
        )
        torch.testing.assert_close(
            bridge.action_efficient_to_aha(efficient_action), action
        )
        state = torch.tensor([[0.5, 4.0]])
        torch.testing.assert_close(
            bridge.state_aha_to_efficient(state), torch.tensor([[1.0, 1.0]])
        )

    def test_stage1_checkpoint_loads_strictly(self) -> None:
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
        original = OVCRSActionGenerator(config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "student.pt"
            torch.save(
                {
                    "format": "ovcr_s_stage1",
                    "student": original.state_dict(),
                    "student_config": config.to_dict(),
                    "step": 12,
                },
                path,
            )
            restored, payload = load_ovcrs_student(
                path,
                device="cpu",
                dtype=torch.float32,
            )
        self.assertEqual(payload["step"], 12)
        for expected, actual in zip(original.parameters(), restored.parameters()):
            torch.testing.assert_close(expected, actual)


if __name__ == "__main__":
    unittest.main()
