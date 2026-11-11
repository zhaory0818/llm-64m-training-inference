"""Derive paired medians, raw tables and exportable figures from completed runs."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import median

from common import ROOT, run_dir, read, dump


def paired(rows, a, b, condition, metric, keys):
    groups = defaultdict(dict)
    for row in rows:
        groups[tuple(row[k] for k in keys)][row[condition]] = row
    values = [cell[b][metric] / cell[a][metric] for cell in groups.values() if a in cell and b in cell]
    assert values
    return {'pairs': len(values), 'median_ratio': median(values),
        'ratio_min': min(values), 'ratio_max': max(values),
        'improvement_percent': (median(values) - 1) * 100}


def grouped(rows, keys, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return [{**dict(zip(keys, key)), 'n': len(cell),
        **{f'{metric}_{stat}': fn([r[metric] for r in cell])
            for metric in metrics for stat, fn in [('median', median), ('min', min), ('max', max)]}}
        for key, cell in sorted(groups.items())]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    folder = args.run_dir or run_dir()
    for name in ['inference_result.json', 'training_result.json', 'cached_sdpa/result.json']:
        assert read(folder / name)['completed']
    train = read(folder / 'training_records.json')
    inference = read(folder / 'inference_records.json')
    sdpa = read(folder / 'cached_sdpa/records.json')
    result = {'training': {}, 'lora': {}, 'sdpa': {}, 'projection': {},
        'metrics_note': 'Paired ratio medians; positive throughput change differs from negative latency change. No quality metric.',
        'weights_unchanged_during_experiments': True}
    for phase in ['development', 'confirmation']:
        cells = [r for r in train if r['phase'] == phase]
        result['training'][phase] = paired(cells, 'fixed768', 'dynamic8', 'variant',
            'effective_input_tokens_per_second', ['repeat'])
        cell = result['training'][phase]
        for key in ['processed_tokens', 'input_tokens', 'target_tokens', 'padding_fraction', 'width_mean',
                    'ms_per_update', 'effective_input_tokens_per_second', 'peak_allocated_bytes', 'peak_reserved_bytes']:
            cell[key] = {variant: median([r[key] for r in cells if r['variant'] == variant])
                for variant in ['fixed768', 'dynamic8']}
        cell['latency_reduction_percent'] = (1 - 1 / cell['median_ratio']) * 100
        cell['processed_positions_reduction_percent'] = (1 - cell['processed_tokens']['dynamic8'] / cell['processed_tokens']['fixed768']) * 100
        cells = [r for r in sdpa if r['phase'] == phase]
        result['sdpa'][phase] = {variant: paired(cells, 'original', variant, 'condition',
            'overall_tokens_per_second', ['repeat', 'input_tokens', 'new_tokens'])
            for variant in ['projection_only', 'cached_sdpa', 'combined']}
        cells = [r for r in inference if r['group'] == ('lora_deployment' if phase == 'development' else 'confirmation')]
        result['lora'][phase] = paired(cells, 'runtime_lora', 'merged_lora', 'condition',
            'overall_tokens_per_second', ['repeat', 'input_tokens', 'new_tokens'])
    result['counts'] = {'inference_initial_timed_records': len(inference), 'inference_sdpa_timed_records': len(sdpa),
        'training_timed_trials': len(train), 'training_timed_updates': sum(r['updates'] for r in train),
        'training_warmup_discarded_updates': sum(r['warmup_updates_discarded'] for r in train)}
    result['tables'] = {
        'training': grouped(train, ['phase', 'variant'], ['ms_per_update', 'effective_input_tokens_per_second', 'peak_allocated_bytes', 'peak_reserved_bytes']),
        'sdpa': grouped(sdpa, ['phase', 'input_tokens', 'new_tokens', 'condition'], ['overall_tokens_per_second', 'prefill_ms', 'decode_ms_per_token']),
        'initial': grouped(inference, ['group', 'input_tokens', 'new_tokens', 'condition'], ['overall_tokens_per_second', 'prefill_ms', 'peak_allocated_bytes'])}
    # Greedy output comparisons are observations, not universal numerical equality.
    for phase in ['development', 'confirmation']:
        cells = defaultdict(dict)
        for row in sdpa:
            if row['phase'] == phase:
                cells[(row['repeat'], row['input_tokens'], row['new_tokens'])][row['condition']] = row
        result['sdpa'][phase]['greedy_exact'] = {variant: {
            'matching': sum(c[variant]['generated_ids'] == c['original']['generated_ids'] for c in cells.values()),
            'total': len(cells)} for variant in ['projection_only', 'cached_sdpa', 'combined']}
    dump(folder / 'summary.json', result)
    for name, rows in result['tables'].items():
        with (folder / f'summary_{name}.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for ax, phase in zip(axes[:2], ['development', 'confirmation']):
        cells = [r for r in train if r['phase'] == phase]
        values = [[r['effective_input_tokens_per_second'] for r in cells if r['variant'] == v] for v in ['fixed768', 'dynamic8']]
        ax.bar(['Fixed 768', 'Dynamic / 8'], [median(v) for v in values], color=['#65758b', '#158b8e'])
        for i, v in enumerate(values): ax.scatter([i] * len(v), v, color='black', s=18)
        ax.set(title=f'SFT: {phase} (batch=4)', ylabel='Effective input tokens / second', ylim=(0, 19000))
    cells = [r for r in sdpa if r['phase'] == 'confirmation']
    labels = ['original', 'projection_only', 'cached_sdpa', 'combined']
    values = [[r['overall_tokens_per_second'] for r in cells if r['condition'] == v] for v in labels]
    axes[2].bar(['Original', 'Last only', 'SDPA', 'Both'], [median(v) for v in values], color=['#65758b', '#99b4bd', '#22819a', '#158b8e'])
    for i, v in enumerate(values): axes[2].scatter([i] * len(v), v, color='black', s=18)
    axes[2].set(title='Inference confirmation: 256 + 64', ylabel='Generated tokens / second')
    fig.suptitle('MiniMind 64M | RTX 3060 Laptop 6 GB | dots = independent repeats')
    fig.savefig(folder / 'efficiency.png', dpi=180)
    fig.savefig(folder / 'efficiency.svg')
    plt.close(fig)
    print(json.dumps({k: v for k, v in result.items() if k != 'tables'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
