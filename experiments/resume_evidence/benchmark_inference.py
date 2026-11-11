"""Fixed-work original-forward benchmarks: projection, cache, and LoRA deployment."""
import csv
import gc
import random
import statistics
import time
from pathlib import Path

from common import ROOT, run_dir, read, dump, sha, initialize_torch, load_model, verify_sources, environment, WEIGHTS


def main():
    folder = run_dir()
    assert not (folder / 'inference_result.json').exists()
    torch = initialize_torch()
    from transformers import AutoTokenizer
    protocol = read(folder / 'protocol.json')['inference']
    dump(folder / 'inference_protocol.json', {**protocol, 'script_sha256': sha(Path(__file__)),
        'recorded_before_execution': True, 'latency_driver': 'Greedy original forward, fixed N tokens, no EOS; includes concat/argmax; no tokenization/loading.'})
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'model', local_files_only=True)
    models = {name: load_model(name, torch.float16).requires_grad_(False) for name in ['base', 'lora', 'merged']}
    active = None

    def activate(name):
        nonlocal active
        if active != name:
            if active is not None:
                models[active].cpu()
                gc.collect()
                torch.cuda.empty_cache()
            models[name].cuda()
            active = name
        return models[name]

    def prompt_ids(length, confirmation=False):
        text = ('团队完成项目后记录步骤，核对数据和配置，复查实验结果。' if confirmation
            else '学习新知识需要理解例子，检查输入与输出，并记录实验过程。')
        tokens = tokenizer(tokenizer.bos_token + text * 160, add_special_tokens=False).input_ids[:length]
        assert len(tokens) == length
        return torch.tensor([tokens], dtype=torch.long, device='cuda')

    @torch.inference_mode()
    def measure(model, ids, n, cache=True, keep=0):
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        generated, past, current = ids, None, ids
        wall = time.perf_counter()
        events[0].record()
        for step in range(n):
            output = model(input_ids=current, past_key_values=past,
                use_cache=cache, logits_to_keep=keep)
            nxt = output.logits[:, -1, :].argmax(-1, keepdim=True)
            generated = torch.cat([generated, nxt], dim=1)
            past = output.past_key_values if cache else None
            current = nxt if cache else generated
            if step == 0:
                events[1].record()
        events[2].record()
        torch.cuda.synchronize()
        wall_seconds = time.perf_counter() - wall
        prefill_ms = events[0].elapsed_time(events[1])
        decode_ms = events[1].elapsed_time(events[2])
        return {'wall_seconds': wall_seconds, 'prefill_ms': prefill_ms,
            'decode_ms_per_token': decode_ms / max(n - 1, 1),
            'decode_tokens_per_second': (n - 1) * 1000 / decode_ms,
            'overall_tokens_per_second': n / wall_seconds,
            'resident_allocated_bytes': baseline,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'transient_peak_bytes': torch.cuda.max_memory_allocated() - baseline,
            'generated_ids': generated[0, ids.shape[1]:].tolist()}

    def differences(left, right):
        left, right = left.float(), right.float()
        delta = left - right
        return {'max_abs': delta.abs().max().item(), 'mean_abs': delta.abs().mean().item(),
            'relative_l2': (delta.norm() / left.norm().clamp_min(1e-12)).item(),
            'last_argmax_equal': torch.equal(left.argmax(-1), right.argmax(-1))}

    checks = []
    with torch.inference_mode():
        model = activate('base')
        for length in [32, 128, 512]:
            ids = prompt_ids(length)
            full = model(input_ids=ids, logits_to_keep=0).logits[:, -1:, :]
            last = model(input_ids=ids, logits_to_keep=1).logits
            diff = differences(full, last)
            assert diff['max_abs'] <= .0625 and diff['relative_l2'] <= .003
            generated = measure(model, ids, 16)
            original = model.generate(input_ids=ids, max_new_tokens=16, eos_token_id=None,
                use_cache=True, do_sample=False, temperature=1., top_p=1., top_k=0,
                repetition_penalty=1., logits_to_keep=0)
            assert original[0, length:].tolist() == generated['generated_ids']
            slim = measure(model, ids, 16, keep=1)
            checks.append({'kind': 'projection', 'length': length, **diff,
                'greedy_16_equal': generated['generated_ids'] == slim['generated_ids'], 'driver_matches_original_generate': True})
        for length in [32, 128, 512]:
            ids = prompt_ids(length)
            lora = activate('lora')
            reference = lora(input_ids=ids, logits_to_keep=1).logits.detach().cpu()
            left = measure(lora, ids, 32)
            merged = activate('merged')
            candidate = merged(input_ids=ids, logits_to_keep=1).logits.detach().cpu()
            right = measure(merged, ids, 32)
            diff = differences(reference, candidate)
            assert diff['mean_abs'] <= .01 and diff['relative_l2'] <= .005
            checks.append({'kind': 'merge', 'length': length, **diff,
                'greedy_32_equal': left['generated_ids'] == right['generated_ids']})
    dump(folder / 'inference_correctness.json', checks)
    del full, last, reference, candidate, original
    gc.collect()
    torch.cuda.empty_cache()

    records = []
    began = time.perf_counter()

    def condition_rows(group, conditions, lengths_and_outputs, confirmation=False):
        cells = [(label, cache, keep, length, output) for length, output in lengths_and_outputs
            for label, cache, keep in conditions]
        for label, cache, keep, length, output in cells:
            model = activate('lora' if label == 'runtime_lora' else 'merged' if label == 'merged_lora' else 'base')
            measure(model, prompt_ids(length, confirmation), output, cache, keep)
        for repeat in range(5):
            schedule = list(cells)
            random.Random(31415 + repeat).shuffle(schedule)
            for label, cache, keep, length, output in schedule:
                model = activate('lora' if label == 'runtime_lora' else 'merged' if label == 'merged_lora' else 'base')
                # On model switches, warm kernels for this loaded representation.
                measure(model, prompt_ids(length, confirmation), 4, cache, keep)
                row = {'group': group, 'condition': label, 'repeat': repeat,
                    'input_tokens': length, 'new_tokens': output, 'cache': cache, 'keep': keep,
                    **measure(model, prompt_ids(length, confirmation), output, cache, keep)}
                records.append(row)
                dump(folder / 'inference_records.json', records)
                print(f'[INFER] {len(records)} {group} {label} T={length}+{output} r={repeat} {row["overall_tokens_per_second"]:.1f} tok/s', flush=True)

    condition_rows('projection_cache', [('default', True, 0), ('last_projection', True, 1), ('no_cache_mechanism', False, 0)],
        [(t, n) for t in [32, 128, 512] for n in [32, 128]])
    condition_rows('lora_deployment', [('runtime_lora', True, 0), ('merged_lora', True, 0)], [(128, 128), (512, 128)])
    condition_rows('confirmation', [('default', True, 0), ('last_projection', True, 1),
        ('runtime_lora', True, 0), ('merged_lora', True, 0)], [(256, 64)], confirmation=True)
    assert len(records) == 130
    with (folder / 'inference_records.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = [key for key in records[0] if key != 'generated_ids']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in records)
    verify_sources(folder)
    dump(folder / 'inference_result.json', {'completed': True, 'records': len(records),
        'seconds': time.perf_counter() - began, 'correctness': checks,
        'weights_and_original_sources_unchanged': True, 'training_performed': False,
        'environment_after': environment(torch),
        'parameter_counts': {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()},
        'file_bytes': {name: (ROOT / path).stat().st_size for name, (path, _) in WEIGHTS.items()}})
    print('[INFERENCE COMPLETE]', folder, flush=True)


if __name__ == '__main__':
    main()
