"""Test this checkout, recording provenance without pretending it is upstream."""
import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
BASE = '56952d10a8ce9c1224e7c639c17fae92dfe78a94'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--require-torch', action='store_true')
    args = parser.parse_args()
    subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', BASE, 'HEAD'], check=True)
    sys.path.insert(0, str(ROOT))
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    failed_tests = {getattr(test, 'test_case', test).id() for test, _ in result.failures+result.errors}
    passed = result.testsRun-len(failed_tests)-len(result.skipped)
    success = result.wasSuccessful() and (not args.require_torch or not result.skipped)
    files = ['lingbotvla/data/vla_data/'+name for name in ['utils.py', 'transform.py', 'ee_pose_transform.py']]
    files += ['deploy/lingbot_vla_v2_policy.py', 'deploy/action_contract.py',
              'external/RoboTwin/policy/lingbot-vla-v2/serve.py', 'ysj_utils/action_experiment.py', 'ysj_utils/evaluate.py',
              'ysj_utils/compare_results.py', 'configs/robot_configs/robotwin.yaml']
    files += [str(path.relative_to(ROOT)).replace('\\', '/') for path in Path(__file__).parent.glob('*.py')]
    files += ['deploy/ee_actions.py','lingbotvla/action_spaces/ee_labels.py','lingbotvla/action_spaces/loss.py',
              'lingbotvla/data/vla_data/base_dataset.py',
              'lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py',
              'lingbotvla/models/vla/lingbot_vla/configuration_lingbot_vla.py',
              'lingbotvla/utils/arguments.py','tasks/vla/train_lingbotvla.py',
              'external/RoboTwin/policy/lingbot-vla-v2/deploy_policy.py','ysj_utils/prepare_ee_labels.py']
    files += ['lingbotvla/action_spaces/spline.py','ysj_utils/check_spline_fit.py']
    packages = {}
    for name in ('torch', 'numpy', 'scipy', 'PyYAML', 'pyarrow', 'h5py'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    report = dict(status=('PASS_WITH_SKIPS' if result.skipped else 'PASS') if success else 'FAIL',
                  passed=passed, tests_run=result.testsRun,
                  skipped=[dict(test=str(t), reason=r) for t, r in result.skipped],
                  failures=[dict(test=str(t), traceback=r) for t, r in result.failures+result.errors],
                  reviewed_base=BASE, head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
                  python=sys.version, packages=packages,
                  source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files},
                  scope='Local contract/config/source tests. No GPU training or RoboTwin rollout.')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    raise SystemExit(0 if success else 1)


if __name__ == '__main__':
    main()
