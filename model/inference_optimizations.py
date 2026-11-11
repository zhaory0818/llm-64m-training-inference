"""Opt-in SDPA for cached single-token inference, preserving original fallbacks."""
from types import MethodType
import torch
from torch.nn import functional as F
from model.model_minimind import Attention, apply_rotary_pos_emb, repeat_kv


def enable_cached_sdpa(model):
    """Patch only eval, cached, single-query, unmasked attention.

    All keys are past/current tokens, so is_causal=False is correct here. Using
    SDPA's upper-left causal mask for a single query would incorrectly keep only
    the first key. Training, prefill, chunked decode and padded masks retain the
    original implementation. State-dict keys and parameter values are unchanged.
    """
    if getattr(model, '_cached_sdpa_enabled', False):
        return sum(isinstance(layer, Attention) for layer in model.modules())
    if not hasattr(F, 'scaled_dot_product_attention'):
        raise RuntimeError('scaled_dot_product_attention is unavailable')
    count = 0
    for layer in model.modules():
        if not isinstance(layer, Attention):
            continue
        original = layer.forward
        layer._cached_sdpa_original_forward = original

        def cached_forward(self, x, position_embeddings, past_key_value=None,
                           use_cache=False, attention_mask=None, original_forward=original):
            if (self.training or not self.flash or not use_cache or past_key_value is None
                    or x.shape[1] != 1 or attention_mask is not None):
                return original_forward(x, position_embeddings, past_key_value, use_cache, attention_mask)
            batch = x.shape[0]
            q = self.q_norm(self.q_proj(x).view(batch, 1, self.n_local_heads, self.head_dim))
            k = self.k_norm(self.k_proj(x).view(batch, 1, self.n_local_kv_heads, self.head_dim))
            v = self.v_proj(x).view(batch, 1, self.n_local_kv_heads, self.head_dim)
            q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
            k = torch.cat((past_key_value[0], k), dim=1)
            v = torch.cat((past_key_value[1], v), dim=1)
            output = F.scaled_dot_product_attention(q.transpose(1, 2),
                repeat_kv(k, self.n_rep).transpose(1, 2), repeat_kv(v, self.n_rep).transpose(1, 2),
                dropout_p=0., is_causal=False)
            output = output.transpose(1, 2).reshape(batch, 1, -1)
            return self.resid_dropout(self.o_proj(output)), (k, v)

        layer.forward = MethodType(cached_forward, layer)
        count += 1
    original_generate = model.generate
    model._cached_sdpa_original_generate = original_generate

    def normalized_generate(self, *args, **kwargs):
        # Normalize a full-valid mask once per request instead of synchronizing
        # CUDA inside each layer/token. Keep actual padded masks untouched.
        positional = list(args)
        mask = kwargs.get('attention_mask', positional[1] if len(positional) > 1 else None)
        if mask is not None and bool(torch.all(mask == 1)):
            if 'attention_mask' in kwargs:
                kwargs['attention_mask'] = None
            elif len(positional) > 1:
                positional[1] = None
        return original_generate(*positional, **kwargs)

    model.generate = MethodType(normalized_generate, model)
    model._cached_sdpa_enabled = True
    return count


def disable_cached_sdpa(model):
    """Restore the bound methods without altering any weights."""
    if not getattr(model, '_cached_sdpa_enabled', False):
        return
    for layer in model.modules():
        if hasattr(layer, '_cached_sdpa_original_forward'):
            layer.forward = layer._cached_sdpa_original_forward
            del layer._cached_sdpa_original_forward
    model.generate = model._cached_sdpa_original_generate
    del model._cached_sdpa_original_generate
    model._cached_sdpa_enabled = False
