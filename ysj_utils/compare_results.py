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
    modes = tuple(report['contract']['mode'] for report in reports)
    allowed = {('joint_absolute','joint_delta'), ('legacy_absolute','joint_delta'),
               ('joint_delta','joint_delta'),
               ('joint_absolute','ee_delta'), ('joint_delta','ee_delta'),
               ('joint_delta','joint_ee_delta'), ('joint_ee_delta','joint_ee_delta')}
    if modes not in allowed:
        raise ValueError('Unsupported action-space comparison')
    if modes == ('joint_delta','joint_delta'):
        defaults=dict(flow_dimension_weights=None,flow_horizon_weights=None,spline_control_points=0,spline_degree=3)
        controls=[report.get('action_supervision') for report in reports]
        if controls[0] != defaults or not controls[1] or controls[1] == defaults:
            raise ValueError('Supervision comparison requires plain JDelta control and weighted/spline candidate')
        for field in ('state_features_sha256','norm_sha256','action_loss','label_manifests'):
            if field not in reports[0] or field not in reports[1] or reports[0][field] != reports[1][field]:
                raise ValueError(f'Supervision comparison mismatch: {field}')
        if not reports[0]['norm_sha256'] or reports[0]['action_loss'].get('mask_action_padding') is not True:
            raise ValueError('Supervision comparison requires shared normalization and padding masks')
    elif reports[0].get('action_supervision') != reports[1].get('action_supervision'):
        raise ValueError('Action-space comparison must not change weighted/spline supervision at the same time')
    if 'ee_delta' in modes or 'joint_ee_delta' in modes:
        for field in ('state_features_sha256','label_manifests','action_loss'):
            if not all(report.get(field) for report in reports):
                raise ValueError(f'EE comparison requires recorded {field}; rerun evaluation with this version')
        if reports[0]['state_features_sha256'] != reports[1]['state_features_sha256'] or reports[0]['label_manifests'] != reports[1]['label_manifests']:
            raise ValueError('EE comparison requires identical state inputs and audited labels')
        if not all(report['action_loss']['mask_action_padding'] for report in reports):
            raise ValueError('EE comparison requires padding-aware loss in both groups')
        weights = [report['action_loss']['joint_ee_aux_weight'] for report in reports]
        if modes == ('joint_ee_delta','joint_ee_delta') and not (weights[0] == 0 and weights[1] > 0):
            raise ValueError('Expected zero-weight JointEE0 control and positive-weight JointEE candidate')
        if modes == ('joint_ee_delta','joint_ee_delta') and (not reports[0].get('norm_sha256') or reports[0]['norm_sha256'] != reports[1].get('norm_sha256')):
            raise ValueError('JointEE0/JointEE ablation requires identical normalization numbers')
        if modes[1] == 'joint_ee_delta' and modes[0] == 'joint_delta' and weights[1] <= 0:
            raise ValueError('JointEE candidate requires positive EE supervision')
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
        rows.append(dict(task=key[0], setting=key[1], baseline=b, candidate=d, difference_pp=100*(d-b)))
    summary = {}
    for case in ('clean','random'):
        selected = [r for r in rows if r['setting']==case]
        summary[case] = dict(baseline=sum(r['baseline'] for r in selected)/7,
                             candidate=sum(r['candidate'] for r in selected)/7,
                             difference_pp=sum(r['difference_pp'] for r in selected)/7)
    return dict(rows=rows, summary=summary, action_modes=modes,
                action_supervision=[r.get('action_supervision') for r in reports],
                limitation='Descriptive single-training-seed comparison. Same eval seed is recorded but scene identity and per-episode pairing still require verification. No significance or guaranteed improvement claim.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-run', type=Path, required=True)
    parser.add_argument('--candidate-run', '--delta-run', dest='delta_run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.baseline_run, args.delta_run)
    with args.output.open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2)
    print(json.dumps(result['summary'],indent=2))


if __name__ == '__main__':
    main()
