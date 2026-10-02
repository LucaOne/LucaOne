"""CPU regression tests; no pretrained weights or network access required.

Run from the repository root:
    PYTHONPATH=src python -m unittest discover -s src/test -p test_rotary_dtype.py -v
"""
import tempfile
import unittest

import torch

from lucaone import LucaGPLMConfig, LucaGPLMForSequenceClassification
from lucaone.modeling_lucaone import (
    LucaGPLMMultiheadAttention,
    LucaGPLMMultiheadAttentionWithSDPA,
    LucaGPLMRotaryEmbedding,
    apply_rotary_pos_emb,
)


class RotaryDtypeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_cached_tables_preserve_input_dtype_and_gradients(self):
        rotary = LucaGPLMRotaryEmbedding(8)
        # Reuse the same FP32 tables across dtype changes at a fixed length.
        for dtype in (torch.float32, torch.bfloat16, torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                q = torch.randn(2, 4, 5, 8, dtype=dtype, requires_grad=True)
                k = torch.randn(2, 4, 5, 8, dtype=dtype, requires_grad=True)
                q_out, k_out = rotary(q, k)
                self.assertEqual(q_out.dtype, dtype)
                self.assertEqual(k_out.dtype, dtype)
                self.assertEqual(rotary._cos_cached.dtype, torch.float32)
                for original, output in ((q, q_out), (k, k_out)):
                    expected = apply_rotary_pos_emb(
                        original.float(), rotary._cos_cached, rotary._sin_cached
                    ).to(dtype)
                    torch.testing.assert_close(output, expected, rtol=0, atol=0)
                (q_out.float().square().mean() + k_out.float().square().mean()).backward()
                for value in (q, k):
                    self.assertIsNotNone(value.grad)
                    self.assertTrue(torch.isfinite(value.grad).all())
                    self.assertGreater(value.grad.abs().sum().item(), 0)

    def test_query_and_key_dtypes_are_restored_independently(self):
        rotary = LucaGPLMRotaryEmbedding(8)
        q = torch.randn(2, 3, 8, dtype=torch.bfloat16)
        k = torch.randn(2, 5, 8, dtype=torch.float32)
        q_out, k_out = rotary(q, k)
        self.assertEqual(q_out.dtype, q.dtype)
        self.assertEqual(k_out.dtype, k.dtype)

    def test_attention_backward_after_warming_fp32_cache(self):
        for dtype in (torch.bfloat16, torch.float16):
            for attention_class, need_head_weights in (
                (LucaGPLMMultiheadAttention, True),
                (LucaGPLMMultiheadAttentionWithSDPA, False),
                (LucaGPLMMultiheadAttentionWithSDPA, True),
            ):
                with self.subTest(dtype=dtype, attention=attention_class.__name__,
                                  fallback=need_head_weights):
                    attention = attention_class(
                        32, 4, self_attention=True, use_rotary_embeddings=True
                    )
                    # Cached tables are plain attributes, so module.to() leaves
                    # an already populated FP32 cache in place.
                    with torch.no_grad():
                        attention(torch.randn(5, 2, 32), need_head_weights=need_head_weights)
                    attention.to(dtype=dtype).train()
                    self.assertEqual(attention.rot_emb._cos_cached.dtype, torch.float32)
                    x = torch.randn(5, 2, 32, dtype=dtype, requires_grad=True)
                    padding = torch.tensor([[False] * 5, [False] * 4 + [True]])
                    output, _ = attention(
                        x, key_padding_mask=padding, need_head_weights=need_head_weights
                    )
                    self.assertEqual(output.dtype, dtype)
                    self.assertTrue(torch.isfinite(output).all())
                    output.float().square().mean().backward()
                    for projection in (attention.q_proj, attention.k_proj, attention.v_proj):
                        grad = projection.weight.grad
                        self.assertIsNotNone(grad)
                        self.assertTrue(torch.isfinite(grad).all())
                        self.assertGreater(grad.abs().sum().item(), 0)

    def test_bf16_pretrained_classifier_training_step(self):
        config = LucaGPLMConfig(
            hidden_size=32, ffn_dim=64, num_hidden_layers=1,
            num_attention_heads=4, task_type="binary_class", num_labels=1,
            classifier_dropout_prob=0.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            LucaGPLMForSequenceClassification(config).save_pretrained(directory)
            model = LucaGPLMForSequenceClassification.from_pretrained(
                directory, torch_dtype=torch.bfloat16, local_files_only=True
            ).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        before = model.classifier.weight.detach().clone()
        output = model(
            input_ids=torch.tensor([[2, 5, 6, 3], [2, 7, 8, 3]]),
            attention_mask=torch.ones(2, 4, dtype=torch.long),
            labels=torch.tensor([0.0, 1.0]),
        )
        self.assertEqual(output.logits.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())
        optimizer.step()
        self.assertFalse(torch.equal(before, model.classifier.weight))


if __name__ == "__main__":
    unittest.main()
