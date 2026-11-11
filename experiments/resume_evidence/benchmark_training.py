"""Matched real-update SFT workloads: fixed width versus right-padding removal."""
import csv
import gc
import math
import statistics
import time
from pathlib import Path

from common import ROOT, run_dir, read, dump, sha, initialize_torch, load_model, verify_sources, environment, WEIGHTS


def main():
    folder = run_dir()
    assert not (folder / 'training_result.json').exists()
    torch = initialize_torch()
    from torch.utils.data import DataLoader, Dataset
    from dataset.batch_collator import DynamicRightPadding
    spec = read(folder / 'protocol.json')['training']
    assert sha(folder / 'frozen_sft.pt') == read(folder / 'data_manifest.json')['frozen_sha256']
    frozen = torch.load(folder / 'frozen_sft.pt', map_location='cpu', weights_only=True)
    trim = DynamicRightPadding(multiple=8)

    class FrozenDataset(Dataset):
        def __init__(self, indices): self.indices = indices
        def __len__(self): return len(self.indices)
        def __getitem__(self, index):
            i = self.indices[index]
            return frozen['input_ids'][i], frozen['labels'][i]

    # Verify the entire frozen corpus, including ignored real context and EOS.
    for i in range(1024):
        x, y = trim([(frozen['input_ids'][i], frozen['labels'][i])])
        n = int(frozen['lengths'][i])
        assert x.shape[1] >= n
        assert torch.equal(y[0, :n], frozen['labels'][i, :n])
        assert int((y[:, 1:] != -100).sum()) == int((frozen['labels'][i, 1:] != -100).sum())
    dump(folder / 'collator_invariants.json', {'samples': 1024, 'effective_prefixes_preserved': True,
        'shifted_target_counts_preserved': True, 'alignment': 8})

    def correctness(dtype, manual):
        lengths = frozen['lengths'].tolist()
        short = next(i for i, n in enumerate(lengths) if 100 <= n <= 180)
        medium = next(i for i, n in enumerate(lengths) if 300 <= n <= 450)
        indices = [short, medium]
        full_x = frozen['input_ids'][indices]
        full_y = frozen['labels'][indices]
        trim_x, trim_y = trim([(full_x[i], full_y[i]) for i in range(2)])
        model = load_model(manual_attention=manual).cuda().train()
        reference_grads = {}
        reference_logits = None
        loss_values = []
        for variant, x, y in [('fixed', full_x, full_y), ('trimmed', trim_x, trim_y)]:
            model.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dtype == 'bfloat16'):
                out = model(input_ids=x.cuda(), labels=y.cuda())
                loss = out.loss
            loss.backward()
            loss_values.append(loss.detach().item())
            if variant == 'fixed':
                reference_logits = out.logits[:, :trim_x.shape[1]].detach().float().clone()
                reference_grads = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
            else:
                valid = torch.arange(trim_x.shape[1], device='cuda')[None, :] < frozen['lengths'][indices].cuda()[:, None]
                delta = (out.logits.float() - reference_logits)[valid]
                logits_mean = delta.abs().mean().item()
                grad_sq, ref_sq = 0., 0.
                for name, p in model.named_parameters():
                    grad_sq += (p.grad - reference_grads[name]).double().square().sum().item()
                    ref_sq += reference_grads[name].double().square().sum().item()
                relative = math.sqrt(grad_sq / max(ref_sq, 1e-30))
                metrics = {'dtype': dtype, 'manual_attention': manual, 'rows': indices,
                    'widths': [768, trim_x.shape[1]], 'losses': loss_values,
                    'loss_abs': abs(loss_values[0] - loss_values[1]),
                    'valid_logits_mean_abs': logits_mean, 'gradient_relative_l2': relative}
        del out, loss, model, reference_logits, reference_grads
        gc.collect(); torch.cuda.empty_cache()
        return metrics

    checks = []
    for dtype, manual in [('float32', True), ('bfloat16', False)]:
        metrics = correctness(dtype, manual)
        checks.append(metrics)
        dump(folder / 'training_correctness.json', checks)
        tolerance = spec['correctness_fp32'] if dtype == 'float32' else spec['correctness_bf16']
        assert metrics['loss_abs'] <= tolerance['loss_abs'], metrics
        assert metrics['gradient_relative_l2'] <= tolerance['grad_rel_l2'], metrics
        if dtype == 'float32': assert metrics['valid_logits_mean_abs'] <= tolerance['logits_mean_abs'], metrics
        print('[CORRECTNESS]', metrics, flush=True)

    def trial(batch, variant, phase, repeat, updates=32, warmup=3):
        offset = 0 if phase == 'development' else 512
        indices = list(range(offset, offset + (updates + warmup) * batch))
        assert max(indices) < offset + 512
        dataset = FrozenDataset(indices)
        loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=0,
            collate_fn=trim if variant == 'dynamic8' else None)
        model = load_model().cuda().train()
        base = torch.load(ROOT / WEIGHTS['base'][0], map_location='cpu', weights_only=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        def step(x, y):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = model(input_ids=x, labels=y)
                loss = output.loss + output.aux_loss
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            return loss.detach(), norm.detach()

        iterator = iter(loader)
        for _ in range(warmup):
            x, y = next(iterator)
            step(x.cuda(), y.cuda())
        torch.cuda.synchronize()
        # Discard warm-up parameter updates while reusing allocated optimizer buffers.
        model.load_state_dict(base, strict=True)
        optimizer.zero_grad(set_to_none=True)
        for state in optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value): value.zero_()
        del base
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        losses, norms, events, widths = [], [], [], []
        input_count, target_count, processed = 0, 0, 0
        torch.cuda.synchronize()
        begin = time.perf_counter()
        for update in range(updates):
            x, y = next(iterator)
            assert (y[:, 1:] != -100).any()
            target_count += int((y[:, 1:] != -100).sum())
            rows = indices[(warmup + update) * batch:(warmup + update + 1) * batch]
            input_count += int(frozen['lengths'][rows].sum())
            processed += x.numel(); widths.append(x.shape[1])
            x, y = x.cuda(), y.cuda()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            loss, norm = step(x, y)
            end.record()
            events.append((start, end)); losses.append(loss); norms.append(norm)
        torch.cuda.synchronize()
        wall = time.perf_counter() - begin
        peak = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        gpu_seconds = sum(a.elapsed_time(b) for a, b in events) / 1000
        loss_numbers = [value.item() for value in losses]
        norm_numbers = [value.item() for value in norms]
        assert all(math.isfinite(value) for value in loss_numbers + norm_numbers)
        assert all(torch.isfinite(p).all().item() for p in model.parameters())
        steps = sorted({int(state['step'].item()) for state in optimizer.state.values()})
        assert steps == [updates]
        original = torch.load(ROOT / WEIGHTS['base'][0], map_location='cpu', weights_only=True)
        max_delta = max((p.detach() - original[name].cuda().float()).abs().max().item()
            for name, p in model.named_parameters())
        assert max_delta > 0
        row = {'phase': phase, 'repeat': repeat, 'variant': variant, 'batch': batch,
            'updates': updates, 'warmup_updates_discarded': warmup, 'adam_steps': steps,
            'samples': updates * batch, 'input_tokens': input_count, 'target_tokens': target_count,
            'processed_tokens': processed, 'padding_fraction': 1 - input_count / processed,
            'width_mean': statistics.mean(widths), 'wall_seconds': wall, 'compute_gpu_seconds': gpu_seconds,
            'ms_per_update': wall * 1000 / updates, 'samples_per_second': updates * batch / wall,
            'effective_input_tokens_per_second': input_count / wall,
            'supervised_tokens_per_second': target_count / wall,
            'peak_allocated_bytes': peak, 'peak_reserved_bytes': reserved,
            'loss_first': loss_numbers[0], 'loss_last': loss_numbers[-1],
            'loss_mean': statistics.mean(loss_numbers), 'losses': loss_numbers,
            'gradient_norms': norm_numbers, 'widths': widths, 'parameter_max_abs_delta': max_delta,
            'finite_state': True}
        del model, optimizer, original, losses, norms, loss, norm, x, y
        gc.collect(); torch.cuda.empty_cache()
        return row

    preflight, chosen = [], None
    for batch in spec['batch_candidates']:
        try:
            for variant in ['fixed768', 'dynamic8']:
                row = trial(batch, variant, 'development', -1, updates=3, warmup=1)
                preflight.append(row)
                dump(folder / 'training_preflight.json', preflight)
            chosen = batch
            break
        except torch.OutOfMemoryError:
            preflight.append({'batch': batch, 'oom': True})
            gc.collect(); torch.cuda.empty_cache()
    assert chosen is not None
    estimated = sum(r['wall_seconds'] / r['updates'] for r in preflight if 'wall_seconds' in r and r['batch'] == chosen) * 32 * 6
    execution = {'batch': chosen, 'alignment': 8, 'updates_per_timed_trial': 32, 'repeats_per_phase': 3,
        'phases': ['development', 'confirmation'], 'timed_trials': 12,
        'timed_updates_total': 384, 'estimated_timed_seconds': estimated,
        'script_sha256': sha(Path(__file__)), 'collator_sha256': sha(ROOT / 'dataset/batch_collator.py'),
        'frozen_before_formal_timing': True,
        'scope': 'Each repeat uses the same fixed rows within phase, independent fresh initialization, warm-up reset; weights not saved or used as improved models.'}
    dump(folder / 'training_execution.json', execution)
    print('[PREFLIGHT]', execution, flush=True)
    began = time.perf_counter()
    records = []
    for phase in ['development', 'confirmation']:
        for repeat in range(3):
            order = ['fixed768', 'dynamic8'] if repeat % 2 == 0 else ['dynamic8', 'fixed768']
            for variant in order:
                row = trial(chosen, variant, phase, repeat)
                records.append(row)
                dump(folder / 'training_records.json', records)
                print(f'[TRAIN BENCH] {phase} r={repeat} {variant} batch={chosen} {row["effective_input_tokens_per_second"]:.1f} effective tok/s; peak={row["peak_allocated_bytes"]/1024**3:.3f} GiB', flush=True)
    assert len(records) == 12
    for phase in ['development', 'confirmation']:
        for repeat in range(3):
            pair = [r for r in records if r['phase'] == phase and r['repeat'] == repeat]
            assert pair[0]['input_tokens'] == pair[1]['input_tokens']
            assert pair[0]['target_tokens'] == pair[1]['target_tokens']
    with (folder / 'training_records.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = [key for key in records[0] if key not in ['losses', 'gradient_norms', 'widths']]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows({k: r[k] for k in fields} for r in records)
    verify_sources(folder)
    dump(folder / 'training_result.json', {'completed': True, 'batch': chosen, 'timed_trials': 12,
        'timed_updates': sum(r['updates'] for r in records), 'seconds': time.perf_counter() - began,
        'correctness': checks, 'weights_and_original_sources_unchanged': True,
        'no_new_quality_model_claimed': True, 'environment_after': environment(torch)})


if __name__ == '__main__':
    main()
