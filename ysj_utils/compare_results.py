"""Descriptive comparison only: aggregate results cannot support a paired test."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ysj_utils.evaluate import scheme, validate_results


def compare(baseline_run, delta_run):
    runs = [Path(baseline_run), Path(delta_run)]
    reports = [json.loads((run/'ysj_evaluation.json').read_text(encoding='utf-8')) for run in runs]
    for run, report in zip(runs, reports):
        if report['status'] != 'COMPLETE' or report['mode'] != 'full':
            raise ValueError('Both runs must be complete full evaluations')
        validate_results(run/'_result.txt', **scheme('full'))
        training = report['training_summary']
        if training['global_batch_size'] != 128 or training['max_steps'] != 10000 or training['enable_fp32'] is not False or training['enable_mixed_precision'] is not True:
            raise ValueError('Expected global BS128 / 10000 steps / bf16 training')
        if Path(report['checkpoint']).parent.name != 'global_step_10000':
            raise ValueError('Compare step-10000 checkpoints, not earlier snapshots')
    if reports[0]['contract']['mode'] not in ('joint_absolute','legacy_absolute') or reports[1]['contract']['mode'] != 'joint_delta':
        raise ValueError('Expected absolute-joint baseline and joint-delta candidate')
    for field in ['tasks','configs','episodes','seed','inference_precision','use_length',
                  'training_summary','initial_model','train_list_sha256','code_commit']:
        if reports[0][field] != reports[1][field]:
            raise ValueError(f'Experiment mismatch: {field}')
    rows = []
    values = []
    for run in runs:
        units = {}
        for line in (run/'_result.txt').read_text(encoding='utf-8').splitlines():
            if line and not line.startswith('#'):
                task, case, count, rate = line.split('\t')
                units[(task,case)] = int(count.split('/')[0])/25
        values.append(units)
    for key in sorted(values[0]):
        b, d = values[0][key], values[1][key]
        rows.append(dict(task=key[0], setting=key[1], baseline=b, joint_delta=d, difference_pp=100*(d-b)))
    summary = {}
    for case in ('clean','random'):
        selected = [r for r in rows if r['setting']==case]
        summary[case] = dict(baseline=sum(r['baseline'] for r in selected)/7,
                             joint_delta=sum(r['joint_delta'] for r in selected)/7,
                             difference_pp=sum(r['difference_pp'] for r in selected)/7)
    return dict(rows=rows, summary=summary,
                limitation='Descriptive single-training-seed comparison. Same eval seed is recorded but scene identity and per-episode pairing still require verification. No significance or guaranteed improvement claim.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-run', type=Path, required=True)
    parser.add_argument('--delta-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.baseline_run, args.delta_run)
    with args.output.open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2)
    print(json.dumps(result['summary'],indent=2))


if __name__ == '__main__':
    main()
