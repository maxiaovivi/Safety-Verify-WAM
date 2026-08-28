from __future__ import annotations

import unittest

import torch

from safety_verify_wam.stage1.ovcr_s import (
    AHAAlignedActionExpert,
    OVCRSActionGenerator,
    OVCRSConfig,
    _apply_rotary_embedding,
)


def _tiny_config(*, architecture: str, num_registers: int) -> OVCRSConfig:
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
        num_registers=num_registers,
        action_architecture=architecture,
        text_context_dim=6,
        time_embedding_dim=4,
        distill_layers=(1,),
    )


class AHAAlignedActionTest(unittest.TestCase):
    def test_aligned_structure_has_exact_action_tokens_and_context_gradients(self) -> None:
        config = _tiny_config(architecture="aha_aligned", num_registers=0)
        student = OVCRSActionGenerator(config)
        self.assertIsInstance(student.action_expert, AHAAlignedActionExpert)
        self.assertTrue(hasattr(student.action_expert.blocks[0], "cross_attn"))
        self.assertTrue(hasattr(student.action_expert.blocks[0], "norm3"))

        outputs = student(
            noisy_action=torch.randn(2, 2, 2),
            action_t=torch.tensor([250.0, 750.0]),
            initial_state=torch.randn(2, 2),
            observation_tokens=torch.randn(2, 3, 2),
            video_kv_cache=(
                {"k": torch.randn(2, 3, 4), "v": torch.randn(2, 3, 4)},
            ),
            action_context=torch.randn(2, 3, 6),
            action_context_mask=torch.tensor(
                [[True, True, False], [True, True, True]]
            ),
            return_trace=True,
        )
        self.assertEqual(tuple(outputs["action_velocity"].shape), (2, 2, 2))
        self.assertEqual(tuple(outputs["action_hidden"].shape), (2, 2, 4))
        self.assertEqual(tuple(outputs["action_responses"][1].shape), (2, 2, 4))

        outputs["action_velocity"].square().mean().backward()
        self.assertGreater(
            float(student.proprio_encoder.weight.grad.abs().sum()), 0.0
        )
        self.assertGreater(
            float(student.action_expert.text_embedding[0].weight.grad.abs().sum()),
            0.0,
        )
        self.assertGreater(
            float(
                student.action_expert.blocks[0].cross_attn.q.weight.grad.abs().sum()
            ),
            0.0,
        )

    def test_aligned_action_requires_task_context_and_no_registers(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not use register"):
            _tiny_config(architecture="aha_aligned", num_registers=1)
        student = OVCRSActionGenerator(
            _tiny_config(architecture="aha_aligned", num_registers=0)
        )
        with self.assertRaisesRegex(ValueError, "requires task context"):
            student(
                noisy_action=torch.randn(1, 2, 2),
                action_t=torch.tensor([500.0]),
                initial_state=torch.randn(1, 2),
                observation_tokens=torch.randn(1, 3, 2),
                video_kv_cache=(
                    {"k": torch.randn(1, 3, 4), "v": torch.randn(1, 3, 4)},
                ),
            )

    def test_action_rope_preserves_position_zero_and_rotates_position_one(self) -> None:
        tensor = torch.ones(1, 2, 1, 4)
        rotated = _apply_rotary_embedding(tensor)
        torch.testing.assert_close(rotated[:, 0], tensor[:, 0])
        self.assertFalse(torch.equal(rotated[:, 1], tensor[:, 1]))

    def test_old_stage1_checkpoint_is_an_explicit_partial_initialization(self) -> None:
        old_student = OVCRSActionGenerator(
            _tiny_config(architecture="efficient_joint", num_registers=1)
        )
        with torch.no_grad():
            old_student.query_encoder.base_queries.fill_(0.25)
            old_student.action_expert.input_encoder.action_encoder[0].weight.fill_(
                0.5
            )
            old_student.action_expert.blocks[0].wan_action_qkv.fill_(0.75)
            old_student.action_expert.decoder.action_head[0].weight.fill_(1.25)
        checkpoint = {
            "format": "ovcr_s_stage1",
            "student": old_student.state_dict(),
            "student_config": old_student.config.to_dict(),
            "step": 36000,
        }

        aligned = OVCRSActionGenerator(
            _tiny_config(architecture="aha_aligned", num_registers=0)
        )
        metadata = aligned.load_stage1_initialization(
            checkpoint, source="step_036000.pt"
        )

        torch.testing.assert_close(
            aligned.query_encoder.base_queries,
            old_student.query_encoder.base_queries,
        )
        torch.testing.assert_close(
            aligned.action_expert.action_encoder.weight,
            old_student.action_expert.input_encoder.action_encoder[0].weight,
        )
        torch.testing.assert_close(
            aligned.action_expert.blocks[0].self_attn.q.weight,
            old_student.action_expert.blocks[0]
            .wan_action_qkv[0]
            .permute(0, 2, 1)
            .reshape(4, 4),
        )
        torch.testing.assert_close(
            aligned.action_expert.head.weight,
            old_student.action_expert.decoder.action_head[0].weight,
        )
        torch.testing.assert_close(
            aligned.action_expert.blocks[0].cross_attn.o.weight,
            torch.zeros_like(aligned.action_expert.blocks[0].cross_attn.o.weight),
        )
        self.assertEqual(metadata["source_step"], 36000)
        self.assertEqual(metadata["source_action_architecture"], "efficient_joint")
        self.assertEqual(metadata["target_action_architecture"], "aha_aligned")
        self.assertIn(
            "action_expert.action_encoder.weight", metadata["mapped_keys"]
        )

    def test_full_aha_action_expert_can_be_structurally_sliced(self) -> None:
        source_config = OVCRSConfig(
            observation_dim=2,
            query_dim=4,
            num_queries=2,
            video_dim=8,
            num_heads=2,
            head_dim=4,
            num_layers=2,
            editor_rank=2,
            state_dim=2,
            action_dim=2,
            action_hidden_dim=6,
            action_ffn_dim=10,
            action_chunk_size=2,
            num_registers=0,
            action_architecture="aha_aligned",
            text_context_dim=6,
            time_embedding_dim=4,
            distill_layers=(1, 2),
        )
        source = AHAAlignedActionExpert(source_config)
        target = OVCRSActionGenerator(
            _tiny_config(architecture="aha_aligned", num_registers=0)
        )
        source_branch = torch.arange(12, dtype=torch.float32).reshape(2, 6)

        metadata = target.load_aha_action_expert_slice(
            source,
            teacher_layer_mapping=(2,),
            teacher_branch_embedding=source_branch,
        )

        self.assertEqual(metadata["source_hidden"], 6)
        self.assertEqual(metadata["target_hidden"], 4)
        self.assertEqual(metadata["teacher_layers"], [2])
        self.assertEqual(
            target.initialization_metadata["effective_action_initialization"],
            "aha_structured_slice",
        )
        self.assertTrue(
            all(torch.isfinite(value).all() for value in target.state_dict().values())
        )

    def test_teacher_proprio_encoder_is_reused_exactly(self) -> None:
        target = OVCRSActionGenerator(
            _tiny_config(architecture="aha_aligned", num_registers=0)
        )
        source = torch.nn.Linear(2, 6)
        with torch.no_grad():
            source.weight.fill_(0.125)
            source.bias.fill_(-0.25)

        target.load_aha_proprio_encoder(source)

        torch.testing.assert_close(target.proprio_encoder.weight, source.weight)
        torch.testing.assert_close(target.proprio_encoder.bias, source.bias)


if __name__ == "__main__":
    unittest.main()
