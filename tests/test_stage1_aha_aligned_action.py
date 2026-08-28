from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from safety_verify_wam.stage1.ovcr_s import (
    AHAAlignedActionExpert,
    OVCRSActionGenerator,
    OVCRSConfig,
    _apply_rotary_embedding,
    sinusoidal_embedding_1d,
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
    def test_aligned_action_matches_the_released_aha_block_order(self) -> None:
        config = _tiny_config(architecture="aha_aligned", num_registers=0)
        student = OVCRSActionGenerator(config).eval()
        noisy_action = torch.randn(1, 2, 2)
        action_t = torch.tensor([375.0])
        initial_state = torch.randn(1, 2)
        task_context = torch.randn(1, 3, 6)
        task_mask = torch.tensor([[True, True, False]])
        observation = torch.randn(1, 3, 2)
        cache = ({"k": torch.randn(1, 3, 4), "v": torch.randn(1, 3, 4)},)
        conditioning = student.prepare_conditioning(observation, cache)

        actual = student.predict_velocity(
            noisy_action,
            action_t,
            initial_state,
            conditioning,
            action_context=task_context,
            action_context_mask=task_mask,
        )["action_velocity"]

        expert = student.action_expert
        self.assertIsInstance(expert, AHAAlignedActionExpert)
        tokens = expert.action_encoder(noisy_action)
        tokens = tokens + student.action_branch_embedding[1].view(1, 1, -1)
        proprio = student.proprio_encoder(initial_state).unsqueeze(1)
        context = expert.text_embedding(torch.cat([task_context, proprio], dim=1))
        context_mask = torch.cat(
            [task_mask, torch.ones(1, 1, dtype=torch.bool)], dim=1
        )
        timestep = action_t[:, None].expand(-1, noisy_action.shape[1])
        time_embedding = expert.time_embedding(
            sinusoidal_embedding_1d(config.time_embedding_dim, timestep)
        )
        time_modulation = expert.time_projection(time_embedding).view(
            1, noisy_action.shape[1], 6, config.action_hidden_dim
        )

        block = expert.blocks[0]
        modulation = (
            block.modulation.unsqueeze(0) + time_modulation
        ).chunk(6, dim=2)
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = [
            value.squeeze(2) for value in modulation
        ]
        normalized = block.norm1(tokens) * (1 + scale_attn) + shift_attn
        query = block.self_attn.norm_q(block.self_attn.q(normalized)).view(
            1, 2, config.num_heads, config.head_dim
        )
        action_key = block.self_attn.norm_k(block.self_attn.k(normalized)).view_as(
            query
        )
        action_value = block.self_attn.v(normalized).view_as(query)

        def released_aha_rope(value: torch.Tensor) -> torch.Tensor:
            positions = torch.arange(value.shape[1], dtype=torch.float64)
            frequencies = 1.0 / (
                10000
                ** (
                    torch.arange(0, value.shape[-1], 2, dtype=torch.float64)
                    / value.shape[-1]
                )
            )
            angles = torch.outer(positions, frequencies)
            pairs = value.float().reshape(*value.shape[:-1], -1, 2)
            first, second = pairs.unbind(-1)
            rotated = torch.stack(
                [
                    first * angles.cos()[None, :, None]
                    - second * angles.sin()[None, :, None],
                    first * angles.sin()[None, :, None]
                    + second * angles.cos()[None, :, None],
                ],
                dim=-1,
            )
            return rotated.flatten(-2).to(value.dtype)

        query = released_aha_rope(query)
        action_key = released_aha_rope(action_key)
        updated = conditioning["updated_cache"][0]
        video_key = updated["k"][:, 0].view(1, -1, 1, 4)
        video_value = updated["v"][:, 0].view_as(video_key)
        mixed = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            torch.cat([video_key, action_key], dim=1).transpose(1, 2),
            torch.cat([video_value, action_value], dim=1).transpose(1, 2),
        ).transpose(1, 2)
        tokens = tokens + block.self_attn.o(mixed.flatten(2)) * gate_attn

        cross_input = block.norm3(tokens)
        cross_query = block.cross_attn.norm_q(
            block.cross_attn.q(cross_input)
        ).view(1, 2, 1, 4)
        cross_key = block.cross_attn.norm_k(block.cross_attn.k(context)).view(
            1, 4, 1, 4
        )
        cross_value = block.cross_attn.v(context).view(1, 4, 1, 4)
        cross_response = F.scaled_dot_product_attention(
            cross_query.transpose(1, 2),
            cross_key.transpose(1, 2),
            cross_value.transpose(1, 2),
            attn_mask=context_mask[:, None, None, :],
        ).transpose(1, 2)
        tokens = tokens + block.cross_attn.o(cross_response.flatten(2))
        ffn_input = block.norm2(tokens) * (1 + scale_ffn) + shift_ffn
        tokens = tokens + block.ffn(ffn_input) * gate_ffn
        expected = expert.head(tokens)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

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
