"""Recompute strict scores from curated raw answers; no model or network needed."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'docs/learning/setup_helpers'))
from basic_diagnosis_data import CASES, RUBRIC, strict_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--answers', type=Path,
        default=ROOT / 'docs/project/evidence/basic_diagnosis/own_sft_answers.json')
    args = parser.parse_args()
    evidence = ROOT / 'docs/project/evidence/basic_diagnosis'
    assert json.loads((evidence / 'cases.json').read_text(encoding='utf-8')) == CASES
    assert json.loads((evidence / 'rubric.json').read_text(encoding='utf-8')) == RUBRIC
    answers = json.loads(args.answers.read_text(encoding='utf-8'))
    cases = {case['id']: case for case in CASES}
    assert len(answers) == len({row['key'] for row in answers}) == 72
    scores = {}
    for row in answers:
        score = strict_score(cases[row['case_id']], row['text'])
        assert score == row['strict_score'], row['key']
        scores[row['key']] = score
    result = {'scope': 'Strict automatic scoring only; developmental diagnosis, not a sealed benchmark.',
        'answers': len(answers), 'unique_cases': len({row['case_id'] for row in answers}),
        'stored_scores_match_recomputation': True, 'profiles': {}}
    for profile in RUBRIC['profiles']:
        result['profiles'][profile] = {}
        for group in ('instruction', 'reasoning', 'context'):
            rows = [row for row in answers if row['profile'] == profile and row['group'] == group]
            assert len(rows) == 6
            result['profiles'][profile][group] = {
                'correct': sum(scores[row['key']] for row in rows), 'n': len(rows)}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
