"""RoboTwin policy adapter for LingBot-VLA-v2 (runs in the sim env, `robotwin-eval`).

The model needs Python 3.12 + flash-attn while RoboTwin/sapien/curobo live in a 3.10 env, so
unlike my-pi05 the model is NOT loaded in-process: `get_model` connects to a websocket server
started by serve.py (see eval_parallel.sh), and `eval` ships observations over the wire. The
observation layout and the early-stop-on-success rollout mirror the repo's own client,
experiment/robotwin/eval_policy_client_lingbotvla.py.
"""

import sys
from pathlib import Path

import numpy as np

# policy/lingbot-vla-v2/deploy_policy.py -> lingbot-vla-v2 repo root (holds deploy/).
# Appended, not prepended, so RoboTwin's own top-level packages keep precedence.
LINGBOT_ROOT = Path(__file__).resolve().parents[4]
if str(LINGBOT_ROOT) not in sys.path:
    sys.path.append(str(LINGBOT_ROOT))

from deploy.websocket_client_policy import WebsocketClientPolicy


class LingbotClient:

    def __init__(self, host, port, robo_name):
        self.robo_name = robo_name
        self.policy = WebsocketClientPolicy(host=host, port=port)

    def reset(self):
        # path_to_pi_model=None: keep the weights the server loaded, only rebuild the
        # feature transform and clear the cached action chunk.
        self.policy.infer(dict(reset=True, robo_name=self.robo_name, path_to_pi_model=None))

    def infer(self, observation):
        return self.policy.infer(observation)


def encode_obs(TASK_ENV, observation):
    return {
        "observation.images.cam_high": observation["observation"]["head_camera"]["rgb"],
        "observation.images.cam_left_wrist": observation["observation"]["left_camera"]["rgb"],
        "observation.images.cam_right_wrist": observation["observation"]["right_camera"]["rgb"],
        "observation.state": observation["joint_action"]["vector"],
        "task": TASK_ENV.get_instruction(),
    }


def get_model(usr_args):
    return LingbotClient(
        host=usr_args.get("host", "127.0.0.1"),
        port=int(usr_args["port"]),
        robo_name=usr_args.get("robo_name", "robotwin"),
    )


def eval(TASK_ENV, model, observation):
    ret = model.infer(encode_obs(TASK_ENV, observation))
    actions = np.asarray(ret["action"])
    if actions.ndim == 1:
        actions = actions[None]
    for action in actions:
        TASK_ENV.take_action(action)
        # Stop executing the chunk as soon as the task is solved; eval_policy.py checks
        # eval_success after we return.
        if TASK_ENV.eval_success:
            break


def reset_model(model):
    model.reset()
