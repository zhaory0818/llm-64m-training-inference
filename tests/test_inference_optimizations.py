"""Correctness of cached attention and its guarded original fallbacks."""
import unittest
import datasets  # Windows Arrow DLL ordering.
import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.inference_optimizations import enable_cached_sdpa, disable_cached_sdpa


class CachedSDPATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.model = MiniMindForCausalLM(MiniMindConfig(hidden_size=64,
            num_hidden_layers=2, vocab_size=64, max_position_embeddings=128)).eval()
        self.ids = torch.tensor([[1, 4, 8, 9, 11]])

    @torch.inference_mode()
    def compare(self, suffix, mask=None, training=False):
        self.model.train(training)
        past = self.model(self.ids, use_cache=True).past_key_values
        expected = self.model(suffix, past_key_values=past, use_cache=True,
            attention_mask=mask).logits
        enable_cached_sdpa(self.model)
        actual = self.model(suffix, past_key_values=past, use_cache=True,
            attention_mask=mask).logits
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    def test_cached_single_query(self):
        self.compare(torch.tensor([[12]]))

    def test_chunked_decode_fallback(self):
        self.compare(torch.tensor([[12, 13]]))

    def test_actual_padding_fallback(self):
        self.compare(torch.tensor([[12]]), torch.tensor([[0, 1, 1, 1, 1, 1]]))

    def test_training_fallback(self):
        self.compare(torch.tensor([[12]]), training=True)

    @torch.inference_mode()
    def test_generation_mask_normalization_and_restore(self):
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        original = self.model.generate(self.ids, max_new_tokens=12, do_sample=False,
            eos_token_id=None, top_k=0, top_p=1., temperature=1.)
        self.assertEqual(enable_cached_sdpa(self.model), 2)
        self.assertEqual(enable_cached_sdpa(self.model), 2)  # Idempotent.
        actual = self.model.generate(self.ids, torch.ones_like(self.ids),
            max_new_tokens=12, do_sample=False, eos_token_id=None,
            top_k=0, top_p=1., temperature=1.)
        self.assertTrue(torch.equal(actual, original))
        disable_cached_sdpa(self.model)
        disable_cached_sdpa(self.model)
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))


if __name__ == '__main__':
    unittest.main()
