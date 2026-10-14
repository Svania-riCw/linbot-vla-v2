"""LingBot-VLA-v2 websocket policy server for RoboTwin eval (runs in the `lingbotvla` env).

Thin wrapper over deploy/lingbot_vla_v2_policy.py. The only thing it adds is pinning the
action/state normalization stats to the ones the checkpoint was *trained* with. The stock
server falls back to configs/robot_configs/<robo_name>.yaml of this repo, whose norm_stats is
the published assets/norm_stats/robotwin.json (clean+randomized) -- not what a run trained
with `data.robot_config_root=.../robot_config_clean` normalized against. Resolution order:
  1. --norm_stats (explicit)
  2. data.norm_stats_file in the run's lingbotvla_cli.yaml
  3. norm_stats of <data.robot_config_root>/<robo_name>.yaml from the same yaml
  4. None -> stock fallback (repo robot config)
"""

import argparse
import os
import sys
from pathlib import Path

import yaml

# policy/lingbot-vla-v2/serve.py -> RoboTwin/policy/.. -> external/RoboTwin -> lingbot-vla-v2
LINGBOT_ROOT = Path(__file__).resolve().parents[4]


def str2bool(v):
    return str(v).lower() in ("1", "true", "yes", "y")


def training_config_path(model_path):
    return Path(os.environ.get("LINGBOT_INFERENCE_CONFIG")
                or Path(model_path).parent.parent.parent / "lingbotvla_cli.yaml")


def resolve_norm_stats(model_path, robo_name):
    with open(training_config_path(model_path)) as f:
        data_cfg = yaml.safe_load(f).get("data", {})
    if data_cfg.get("norm_stats_file"):
        return data_cfg["norm_stats_file"], "data.norm_stats_file"
    root = data_cfg.get("robot_config_root")
    robot_yaml = Path(root) / f"{robo_name}.yaml" if root else None
    if robot_yaml is not None and robot_yaml.is_file():
        with open(robot_yaml) as f:
            norm = yaml.safe_load(f).get("norm_stats")
        if norm:
            return norm, f"{robot_yaml}"
    return None, "repo configs/robot_configs fallback"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help=".../checkpoints/global_step_N/hf_ckpt")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9330)
    parser.add_argument("--robo_name", default="robotwin")
    parser.add_argument("--norm_stats", default=None)
    parser.add_argument("--use_length", type=int, default=50)
    parser.add_argument("--use_fp32", type=str2bool, default=True)
    parser.add_argument("--use_compile", type=str2bool, default=False)
    args = parser.parse_args()

    norm_stats, source = (args.norm_stats, "--norm_stats") if args.norm_stats else \
        resolve_norm_stats(args.model_path, args.robo_name)
    print(f"[serve] model      : {args.model_path}", flush=True)
    print(f"[serve] norm_stats : {norm_stats}  (from {source})", flush=True)
    print(f"[serve] precision  : {'fp32' if args.use_fp32 else 'bf16'}  compile={args.use_compile}", flush=True)

    # The stock server resolves configs/robot_configs/<robo_name>.yaml relative to cwd.
    os.chdir(LINGBOT_ROOT)
    sys.path.insert(0, str(LINGBOT_ROOT))
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
    from deploy.websocket_policy_server import WebsocketPolicyServer

    model = LingbotVLAv2Server(
        args.model_path,
        robot_norm_path=norm_stats,
        use_length=args.use_length,
        chunk_ret=True,
        use_bf16=not args.use_fp32,
        use_fp32=args.use_fp32,
        use_compile=args.use_compile,
    )
    print(f"[serve] listening on {args.host}:{args.port}", flush=True)
    WebsocketPolicyServer(model, host=args.host, port=args.port).serve_forever()


if __name__ == "__main__":
    main()
