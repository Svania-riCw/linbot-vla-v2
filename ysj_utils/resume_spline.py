"""Continue the completed Spline 5k DCP for another 5k, then evaluate 350 episodes."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ysj_utils.resume_joint_delta import main


def command_defaults():
    return ['--experiment', 'spline_absolute',
            '--source-run', '/2025233210/ysj/lingbot_ws/outputs/baseline_supervision_5k_aligned_20261018/Spline/train',
            '--output-dir', '/2025233210/ysj/lingbot_ws/outputs/spline_5k_to_10k_20261019',
            '--additional-steps', '5000', '--inference-compile', 'false']


if __name__ == '__main__':
    sys.argv[1:1] = command_defaults()
    main()
