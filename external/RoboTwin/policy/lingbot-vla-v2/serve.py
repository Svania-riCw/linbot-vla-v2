"""LingBot-VLA-v2 websocket policy server for RoboTwin eval.

Pins both robot action representation and norm statistics to the checkpoint's
training config. Delta checkpoints also validate a norm provenance receipt.
An explicitly configured but missing training robot root is an error.
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

    sys.path.insert(0, str(LINGBOT_ROOT))
    from deploy.action_contract import resolve_contract
    contract = resolve_contract(args.model_path, args.robo_name, args.norm_stats)
    norm_stats, source = contract['norm_stats'], contract['config']
    print(f"[serve] model      : {args.model_path}", flush=True)
    print(f"[serve] norm_stats : {norm_stats}  (from {source})", flush=True)
    print(f"[serve] robot config: {contract['robot_config']}  mode={contract['mode']}", flush=True)
    print(f"[serve] precision  : {'fp32' if args.use_fp32 else 'bf16'}  compile={args.use_compile}", flush=True)

    # The stock server resolves configs/robot_configs/<robo_name>.yaml relative to cwd.
    os.chdir(LINGBOT_ROOT)
    sys.path.insert(0, str(LINGBOT_ROOT))
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
    from deploy.websocket_policy_server import WebsocketPolicyServer

    model = LingbotVLAv2Server(
        args.model_path,
        robot_norm_path=norm_stats,
        robot_config_root=contract['robot_config_root'],
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
