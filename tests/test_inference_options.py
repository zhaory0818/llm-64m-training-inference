import unittest
import datasets
import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.inference_options import configure_inference


class OptionsTests(unittest.TestCase):
    def test_options_propagate_and_explicit_override_survives(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        model = MiniMindForCausalLM(MiniMindConfig(hidden_size=64, num_hidden_layers=1,
            vocab_size=64, max_position_embeddings=64)).eval()
        ids = torch.tensor([[1, 4, 7]])
        args = dict(max_new_tokens=5, do_sample=False, eos_token_id=None, top_k=0,
            top_p=1., temperature=1.)
        expected = model.generate(ids, **args)
        configure_inference(model, last_logits=True, cached_sdpa=True)
        widths = []
        hook = model.lm_head.register_forward_pre_hook(lambda _, inputs: widths.append(inputs[0].shape[1]))
        actual = model.generate(ids, attention_mask=torch.ones_like(ids), **args)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(widths, [1] * 5)
        widths.clear()
        model.generate(ids, logits_to_keep=0, **args)
        self.assertEqual(widths[0], 3)
        hook.remove()


if __name__ == '__main__':
    unittest.main()
