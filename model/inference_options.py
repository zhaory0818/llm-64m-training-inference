"""Explicit generation options shared by the CLI and HTTP/SSE demo."""
from types import MethodType
from model.inference_optimizations import enable_cached_sdpa


def configure_inference(model, *, last_logits=False, cached_sdpa=False):
    if cached_sdpa:
        enable_cached_sdpa(model)
    if last_logits and not getattr(model, '_last_logits_enabled', False):
        original = model.generate

        def generate_last(self, *args, **kwargs):
            kwargs.setdefault('logits_to_keep', 1)
            return original(*args, **kwargs)

        model.generate = MethodType(generate_last, model)
        model._last_logits_enabled = True
    return model
