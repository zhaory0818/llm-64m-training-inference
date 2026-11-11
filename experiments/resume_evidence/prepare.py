"""Freeze actual SFT examples and experiment protocols without training."""
from datetime import datetime
import argparse
import json
import random
import shutil
import time

from common import ROOT, POINTER, WEIGHTS, dump, sha, read, initialize_torch, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', type=__import__('pathlib').Path,
        help='Official SFT JSONL; deterministic reservoir sampling without prior audit files.')
    args = parser.parse_args()
    folder = ROOT / 'logs/resume_evidence' / f'run_{datetime.now():%Y%m%d_%H%M%S}'
    folder.mkdir(parents=True, exist_ok=False)
    dump(POINTER, {'folder': str(folder)})
    source_files = ['model/model_minimind.py', 'model/model_lora.py', 'dataset/lm_dataset.py',
        'trainer/train_full_sft.py', 'trainer/trainer_utils.py', 'model/tokenizer.json',
        'model/tokenizer_config.json', 'tools/project_demo.py']
    before = {p: sha(ROOT / p) for p in source_files}
    for path, expected in WEIGHTS.values():
        assert sha(ROOT / path) == expected
        before[path] = expected
    dump(folder / 'protected_before.json', before)
    for rel in source_files:
        target = folder / 'source_before' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    protocol = {'frozen_at': datetime.now().astimezone().isoformat(),
        'scope': 'Bounded local efficiency/correctness experiments; no task-quality improvement claim.',
        'hypotheses': ['Last-position projection reduces unused prefill projection',
            'Merged LoRA reduces runtime branch overhead', 'Right-padding removal reduces training waste'],
        'inference': {'prompt_tokens': [32, 128, 512], 'new_tokens': [32, 128], 'repeats': 5,
            'warmup_per_condition': 1, 'dtype': 'float16', 'batch': 1, 'EOS_disabled_for_equal_work': True,
            'primary_baseline': 'cache=True, logits_to_keep=0',
            'projection_candidate': 'cache=True, logits_to_keep=1',
            'mechanism_only': 'cache=False, logits_to_keep=0',
            'confirmation': {'prompt_tokens': 256, 'new_tokens': 64, 'repeats': 5, 'different_text': True},
            'measurement': 'Independent greedy driver calls original forward; GPU-event prefill/decode and synchronized wall time; not HTTP TTFT.',
            'numerical_tolerance': {'projection_fp16_max_abs': .0625, 'projection_fp16_rel_l2': .003,
                'merge_fp16_mean_abs': .01, 'merge_fp16_rel_l2': .005},
            'greedy_equality': 'Report separately; approximate FP16 logit agreement is not universal exact text equivalence.'},
        'training': {'sample_count': 1024, 'split': '512 development + 512 confirmation; performance workloads, not quality validation',
            'sample_seed': 20261031, 'preprocessing_seed': '20261031 + original row', 'max_seq_len': 768,
            'batch_candidates': [4, 2, 1], 'dtype': 'bfloat16 autocast / FP32 parameters',
            'short_preflight_updates': 3, 'formal_updates_per_repeat': 32, 'repeats': 3,
            'confirmation_updates': 32, 'confirmation_repeats': 3, 'warmup_updates': 3,
            'optimizer': 'AdamW lr=1e-5, default weight_decay=0.01; clip_grad_norm=1; accumulation=1',
            'comparison': 'Same examples, ordering, initial state, supervision and updates; trim only common unused suffix.',
            'correctness_fp32': {'loss_abs': 2e-5, 'logits_mean_abs': 2e-5, 'grad_rel_l2': 1e-4},
            'correctness_bf16': {'loss_abs': .03, 'grad_rel_l2': .08},
            'no_packing_no_reordering': True,
            'metrics': ['effective input token/s', 'supervised target token/s', 'samples/s', 'ms/update', 'peak allocated/reserved', 'padding fraction']},
        'weights_read_only': True, 'paid_services': False}
    dump(folder / 'protocol.json', protocol)
    torch = initialize_torch()
    dump(folder / 'environment.json', environment(torch))
    from transformers import AutoTokenizer
    from dataset.lm_dataset import SFTDataset
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'model', local_files_only=True)
    if args.data_path:
        source_raw = args.data_path.resolve()
        raw, reservoir_rng = [], random.Random(314159)
        with source_raw.open(encoding='utf-8') as stream:
            for row, line in enumerate(stream):
                if len(raw) < 2048:
                    raw.append({'row': row, 'sample': json.loads(line)})
                else:
                    index = reservoir_rng.randrange(row + 1)
                    if index < 2048:
                        raw[index] = {'row': row, 'sample': json.loads(line)}
        assert len(raw) >= 1024, 'At least 1024 rows required.'
    else:
        audit = __import__('pathlib').Path(read(ROOT / 'docs/learning/generated/basic_diagnosis_20261031/latest_sft_audit.json')['folder'])
        source_raw = audit / 'sampled_raw.json'
        raw = read(source_raw)
    rng = random.Random(20261031)
    rng.shuffle(raw)
    selected = raw[:1024]
    dataset = object.__new__(SFTDataset)
    dataset.max_length = 768
    dataset.tokenizer = tokenizer
    dataset.samples = [item['sample'] for item in selected]
    dataset.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
    dataset.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
    inputs, labels, lengths = [], [], []
    for i, item in enumerate(selected):
        random.seed(20261031 + item['row'])
        x, y = dataset[i]  # The original Dataset implementation is the authority.
        assert x.shape == y.shape == (768,)
        occupied = (x != 0) | (y != -100)
        length = int(torch.where(occupied)[0][-1]) + 1
        assert (y[1:] != -100).any()
        assert torch.all(y[x == 0] == -100), ('supervised PAD', item['row'])
        inputs.append(x); labels.append(y); lengths.append(length)
    payload = {'input_ids': torch.stack(inputs), 'labels': torch.stack(labels),
        'lengths': torch.tensor(lengths), 'rows': torch.tensor([item['row'] for item in selected])}
    torch.save(payload, folder / 'frozen_sft.pt')
    dump(folder / 'data_manifest.json', {'source_file': str(source_raw), 'source_raw_sha256': sha(source_raw),
        'frozen_sha256': sha(folder / 'frozen_sft.pt'), 'rows': [item['row'] for item in selected],
        'samples': 1024, 'length_min': min(lengths), 'length_mean': sum(lengths) / len(lengths),
        'length_max': max(lengths), 'supervised_targets': int((payload['labels'][:, 1:] != -100).sum()),
        'semantics': 'Original SFTDataset fixed inputs/labels; PAD=0 verified nonsupervised; no full Arrow cache rebuilt.'})
    print('RUN_FOLDER', folder, flush=True)
    print('FROZEN_DATA', {'samples': 1024, 'mean_length': sum(lengths)/len(lengths)}, flush=True)


if __name__ == '__main__':
    main()
