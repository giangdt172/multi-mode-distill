import unittest
from types import SimpleNamespace

import torch

from src.batching import pack_trajectories
from src.menger import menger_loss_per_response
from src.modes import menger_enabled_for_mode


class FastByteTokenizer:
    is_fast = True
    eos_token_id = 256
    pad_token_id = 257

    def encode(self, text, add_special_tokens=False):
        return list(text.encode("ascii"))

    def decode(
        self,
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        return bytes(token_ids).decode("ascii")

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        encoded = {"input_ids": self.encode(text)}
        if return_offsets_mapping:
            encoded["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return encoded


class MengerLossTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FastByteTokenizer()
        self.response = torch.tensor(
            self.tokenizer.encode("one\n\ntwo\n\nthree\n\nfour")
            + [self.tokenizer.eos_token_id]
        )

    def _batch(self, prompt):
        return pack_trajectories(
            [torch.tensor(self.tokenizer.encode(prompt))],
            [self.response],
            self.tokenizer.pad_token_id,
            "qwen",
            128,
            "cpu",
        )

    def test_all_three_modes_are_enabled(self):
        args = SimpleNamespace(menger_weight=1.0)
        for mode in ("off_policy", "on_policy", "self_distill"):
            self.assertTrue(menger_enabled_for_mode(args, mode))
        self.assertFalse(menger_enabled_for_mode(args, "opsd"))
        self.assertFalse(
            menger_enabled_for_mode(SimpleNamespace(menger_weight=0.0), "off_policy")
        )

    def test_different_reference_prompt_aligns_and_backpropagates(self):
        student_batch, student_meta = self._batch("Question:")
        reference_batch, reference_meta = self._batch(
            "Question:\n\nAdditional context:\nverified steps"
        )
        torch.manual_seed(7)
        student_hidden = torch.randn(
            1, student_batch["input_ids"].shape[1], 5, requires_grad=True
        )
        teacher_hidden = torch.randn(
            1, reference_batch["input_ids"].shape[1], 9, requires_grad=True
        )

        loss = menger_loss_per_response(
            student_hidden,
            teacher_hidden,
            student_batch,
            student_meta["label"],
            reference_batch,
            reference_meta["label"],
            self.tokenizer,
        ).mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(student_hidden.grad).all())
        self.assertGreater(student_hidden.grad.abs().sum().item(), 0)
        self.assertIsNone(teacher_hidden.grad)

    def test_fewer_than_three_steps_returns_differentiable_zero(self):
        response = torch.tensor(
            self.tokenizer.encode("one\n\ntwo") + [self.tokenizer.eos_token_id]
        )
        batch, metadata = pack_trajectories(
            [torch.tensor(self.tokenizer.encode("Question:"))],
            [response],
            self.tokenizer.pad_token_id,
            "qwen",
            64,
            "cpu",
        )
        student_hidden = torch.randn(
            1, batch["input_ids"].shape[1], 4, requires_grad=True
        )
        teacher_hidden = torch.randn(1, batch["input_ids"].shape[1], 6)
        loss = menger_loss_per_response(
            student_hidden,
            teacher_hidden,
            batch,
            metadata["label"],
            batch,
            metadata["label"],
            self.tokenizer,
        ).mean()
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertEqual(student_hidden.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
