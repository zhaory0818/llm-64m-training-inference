"""Post-hoc hypothesis prompted by cache slowdown; fresh factorial A/B and confirmation."""
import csv
import gc
import random
import time
from pathlib import Path

from common import ROOT, run_dir, dump, sha, initialize_torch, load_model, verify_sources, environment


def main():
    folder = run_dir() / 'cached_sdpa'
    folder.mkdir(exist_ok=False)
    dump(folder / 'protocol.json', {'hypothesis_origin': 'Initial cache-on path was slower than no-cache on this GPU; code inspection found manual single-token attention.',
        'exploratory_extension': True, 'frozen_before_new_timing': True,
        'main_baseline': 'Original cache=True, projection=all; no artificial cache-off baseline',
        'factors': ['original vs cached single-token SDPA route', 'all vs last output projection'],
        'prompt_lengths': [32, 128, 512], 'output_lengths': [32, 128], 'repeats': 5,
        'confirmation': {'length': 256, 'output': 64, 'repeats': 5, 'different_text': True},
        'numerical_limits': {'fp32_max_abs': .001, 'fp32_relative': .0001,
            'fp16_max_abs': .0625, 'fp16_relative': .003},
        'greedy_exact_equality_reported_separately': True, 'training_performed': False,
        'script_sha256': sha(Path(__file__)), 'optimization_sha256': sha(ROOT / 'model/inference_optimizations.py')})
    torch = initialize_torch()
    from transformers import AutoTokenizer
    from model.inference_optimizations import enable_cached_sdpa, disable_cached_sdpa
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'model', local_files_only=True)

    def ids(length, confirmation=False):
        text = '核对记录与数据，验证每一步计算，再整理可复现的实验。' if confirmation else '学习新知识需要理解例子，检查输入与输出，并记录实验过程。'
        tokens = tokenizer(tokenizer.bos_token + text * 160, add_special_tokens=False).input_ids[:length]
        return torch.tensor([tokens], dtype=torch.long, device='cuda')

    # Compare decode logits along the same teacher-forced prefixes.
    checks = []
    for dtype in [torch.float32, torch.float16]:
        model = load_model(dtype=dtype).cuda().eval().requires_grad_(False)
        keys_before = list(model.state_dict())
        for length in [32, 128, 512]:
            prefix = ids(length)
            future = ids(32, True)
            snapshots = []
            with torch.inference_mode():
                for enabled in [False, True]:
                    enable_cached_sdpa(model) if enabled else disable_cached_sdpa(model)
                    past, current, values = None, prefix, []
                    for step in range(17):
                        output = model(input_ids=current, past_key_values=past, use_cache=True, logits_to_keep=1)
                        values.append(output.logits[:, -1, :].float().cpu())
                        past = output.past_key_values
                        current = future[:, step:step + 1]
                    snapshots.append(torch.stack(values))
            a, b = snapshots
            maximum = (a - b).abs().max().item()
            relative = ((a - b).norm() / a.norm()).item()
            assert maximum <= (.001 if dtype == torch.float32 else .0625)
            assert relative <= (.0001 if dtype == torch.float32 else .003)
            checks.append({'dtype': str(dtype), 'length': length, 'teacher_forced_positions': 17,
                'max_abs': maximum, 'relative_l2': relative,
                'argmax_equal': torch.equal(a.argmax(-1), b.argmax(-1))})
        # Real padding must take the original fallback, not a wrong causal shortcut.
        masked = ids(32); mask = torch.ones_like(masked); mask[:, :3] = 0
        with torch.inference_mode():
            enable_cached_sdpa(model)
            a = model.generate(masked, attention_mask=mask, max_new_tokens=16, eos_token_id=None,
                do_sample=False, temperature=1., top_p=1., top_k=0)
            disable_cached_sdpa(model)
            b = model.generate(masked, attention_mask=mask, max_new_tokens=16, eos_token_id=None,
                do_sample=False, temperature=1., top_p=1., top_k=0)
            assert torch.equal(a, b)
        assert list(model.state_dict()) == keys_before
        checks.append({'dtype': str(dtype), 'padded_mask_fallback_exact': True, 'state_keys_unchanged': True})
        del model, snapshots, output, past, current
        gc.collect(); torch.cuda.empty_cache()
        dump(folder / 'correctness.json', checks)
    model = load_model(dtype=torch.float16).cuda().eval().requires_grad_(False)

    @torch.inference_mode()
    def measure(prefix, steps, keep):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start, first, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        generated, current, past = prefix, prefix, None
        began = time.perf_counter(); start.record()
        for step in range(steps):
            output = model(input_ids=current, past_key_values=past, use_cache=True, logits_to_keep=keep)
            nxt = output.logits[:, -1, :].argmax(-1, keepdim=True)
            generated = torch.cat([generated, nxt], dim=1)
            past, current = output.past_key_values, nxt
            if step == 0: first.record()
        end.record(); torch.cuda.synchronize()
        seconds = time.perf_counter() - began
        decode_ms = first.elapsed_time(end)
        return {'wall_seconds': seconds, 'prefill_ms': start.elapsed_time(first),
            'decode_ms_per_token': decode_ms / max(steps - 1, 1),
            'decode_tokens_per_second': (steps - 1) * 1000 / decode_ms,
            'overall_tokens_per_second': steps / seconds,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'generated_ids': generated[0, prefix.shape[1]:].tolist()}

    # Operator evidence is collected outside the timed runs.
    profiles = []
    with torch.inference_mode():
        for enabled in [False, True]:
            enable_cached_sdpa(model) if enabled else disable_cached_sdpa(model)
            initial = model(input_ids=ids(128), use_cache=True, logits_to_keep=1)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
                model(input_ids=ids(1), past_key_values=initial.past_key_values, use_cache=True, logits_to_keep=1)
            profiles.append({'enabled': enabled, 'operators': {item.key: item.count for item in profiler.key_averages()
                if any(word in item.key for word in ['scaled_dot_product', 'softmax', 'bmm'])}})
    dump(folder / 'operator_profile.json', profiles)
    del initial

    rows = []
    began = time.perf_counter()
    conditions = [('original', False, 0), ('projection_only', False, 1), ('cached_sdpa', True, 0), ('combined', True, 1)]
    for phase, cells in [('development', [(t, n) for t in [32, 128, 512] for n in [32, 128]]),
                         ('confirmation', [(256, 64)])]:
        for length, output in cells:
            for _, enabled, keep in conditions:
                enable_cached_sdpa(model) if enabled else disable_cached_sdpa(model)
                measure(ids(length, phase == 'confirmation'), output, keep)
        for repeat in range(5):
            schedule = [(length, output, label, enabled, keep) for length, output in cells
                for label, enabled, keep in conditions]
            random.Random(9821 + repeat).shuffle(schedule)
            for length, output, label, enabled, keep in schedule:
                enable_cached_sdpa(model) if enabled else disable_cached_sdpa(model)
                row = {'phase': phase, 'repeat': repeat, 'condition': label,
                    'input_tokens': length, 'new_tokens': output,
                    **measure(ids(length, phase == 'confirmation'), output, keep)}
                rows.append(row); dump(folder / 'records.json', rows)
                print(f'[CACHED SDPA] {len(rows)} {phase} {label} {length}+{output} r={repeat} {row["overall_tokens_per_second"]:.1f} tok/s', flush=True)
    assert len(rows) == 140
    with (folder / 'records.csv').open('w', encoding='utf-8', newline='') as stream:
        fields = [key for key in rows[0] if key != 'generated_ids']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    disable_cached_sdpa(model)
    verify_sources(run_dir())
    dump(folder / 'result.json', {'completed': True, 'records': len(rows), 'correctness': checks,
        'seconds': time.perf_counter() - began, 'training_performed': False,
        'original_sources_and_weights_unchanged': True, 'environment_after': environment(torch)})


if __name__ == '__main__':
    main()
