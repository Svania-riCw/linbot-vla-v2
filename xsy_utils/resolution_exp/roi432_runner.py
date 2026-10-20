"""Run one prepared ROI-432 experiment on its verified host, without a GPU queue.

Smoke and formal are separate invocations: formal requires reviewed smoke evidence.
No failed phase is retried, no existing output is deleted, and no other job is stopped.
"""
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time

import yaml


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n")
    temporary.replace(path)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def validate_recipe(config, source):
    """Allow only the declared input representation and run bookkeeping changes."""
    required = {
        "data": dict(img_size=384, image_resize_mode="opencv_lanczos4",
                     preserve_image_resolution=True, num_workers=8,
                     prefetch_factor=4, pin_memory=True),
        "train": dict(global_batch_size=128, micro_batch_size=4,
                      gradient_accumulation_steps=4, max_steps=10000,
                      save_steps=1000, seed=42, enable_fp32=True,
                      pause_after_steps=5000, enable_resume=False,
                      allow_dataloader_worker_change=False),
    }
    for section, expected in required.items():
        for key, value in expected.items():
            if config[section].get(key) != value:
                raise ValueError(f"Locked experiment setting changed: {section}.{key}")
    if config["train"].get("load_checkpoint_path"):
        raise ValueError("A new formal experiment must start at the original initialization")
    allowed = {
        "data": {"roi_config", "roi_geometry_root"},
        "train": {"output_dir", "wandb_name", "enable_resume", "pause_after_steps",
                  "allow_dataloader_worker_change", "load_checkpoint_path"},
    }
    for section in set(source) | set(config):
        left, right = source.get(section, {}), config.get(section, {})
        if not isinstance(left, dict) or not isinstance(right, dict):
            if left != right:
                raise ValueError(f"Changed recipe section: {section}")
            continue
        for key in set(left) | set(right):
            if key not in allowed.get(section, set()) and left.get(key) != right.get(key):
                raise ValueError(f"Undeclared comparison change: {section}.{key}")
    roi = config["data"].get("roi_config", {})
    if roi.get("mode") != "ee" or roi.get("image_size") != 384:
        raise ValueError("Expected the prepared EE ROI-432 recipe")


def alive(pid):
    try:
        return Path("/proc", str(pid), "stat").read_text().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
    except FileNotFoundError:
        return False


def main():
    import fcntl  # Linux-only process lock; recipe validation also runs on Windows.

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    repo, job = Path(plan["repo"]).resolve(), Path(plan["job"]).resolve()
    state_dir = job / "engine_state"
    config_path = Path(plan["config"])
    config = yaml.safe_load(config_path.read_text())
    source = yaml.safe_load(Path(plan["reference_config"]).read_text())
    validate_recipe(config, source)
    assert hashlib.sha256(Path(plan["reference_config"]).read_bytes()).hexdigest() == plan["reference_config_sha256"]
    assert socket.gethostname() == plan["hostname"]
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    assert head == plan["head"]
    assert not subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == plan["config_sha256"]
    assert read(job / "inputs/verified_preflight.json")["ready"] is True
    assert Path(plan["ipc_so"]).is_file()
    ipc = Path(plan["ipc_directory"])
    assert ipc.is_dir() and ipc.stat().st_mode & 0o077 == 0
    lock = (repo / ".runtime/priority_worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ.update(QL_REPO=str(repo), QL_QUEUE_STATE_DIR=str(state_dir),
                      QL_OUTPUT_ROOT=str(job / "outputs"), QL_EXPECTED_HEAD=head)
    spec = importlib.util.spec_from_file_location("roi_engine", repo / "xsy_utils/resolution_exp/queue_engine.py")
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)

    def idle(current=None, deadline=None):
        # Thirty seconds of consecutive real idleness; fail immediately on a job.
        for check in range(3):
            cards, apps = engine.gpu_snapshot()
            workloads = [w for w in engine.active_gpu_workloads() if alive(w["pid"])]
            if apps.strip() or workloads or not engine.snapshot_idle(cards, apps):
                raise RuntimeError("GPUs are occupied or not released; no queue, restart or interference")
            if check < 2:
                time.sleep(15)

    engine.wait_idle = idle
    idle()
    claim = job / "claim"
    if args.mode == "smoke":
        claim.mkdir()  # Never remove or silently reuse a failed claim.
        owner = dict(node="node1", hostname=plan["hostname"], pid=os.getpid(), claimed_utc=utc())
    else:
        owner = read(claim / "owner.json")
        assert owner["hostname"] == plan["hostname"] and owner["node"] == "node1"
        assert owner.get("result") == "smoke_complete" and not alive(owner["pid"])
        review = read(job / "inputs/verified_smoke.json")
        assert review["ready"] is True and review["config_sha256"] == plan["config_sha256"]
        owner.update(previous_pid=owner["pid"], pid=os.getpid(), result=None, formal_started_utc=utc())
    atomic(claim / "owner.json", owner)
    state = dict(pid=os.getpid(), owner_node="node1", head=head, experiment="roi432_ee",
                 mode=args.mode, job=str(job), config=str(config_path), started_utc=utc())
    logging = read(state_dir / "wandb_settings.json")
    assert logging["entity"] == "qlei34151-shanghaitech-university"
    assert logging["project"] == "lingbot-vla-roi"
    secret = Path(logging["secret_file"])
    assert secret.stat().st_mode & 0o077 == 0
    (state_dir / "wandb").mkdir(exist_ok=True)
    experiment = plan["experiment_id"]

    def train(smoke, resume):
        phase = ("smoke_resume" if resume else "smoke") if smoke else (
            "resume_to_10000" if resume else "train_to_5000")
        target = (4 if resume else 2) if smoke else (10000 if resume else 5000)
        resume_step = (2 if smoke else 5000) if resume else 0
        run_name = f'{experiment}_{"smoke" if smoke else "train"}_b1_384_m4'
        output = job / "outputs" / run_name
        if resume:
            engine.checkpoint_valid(output, resume_step)
            before = state_dir / (phase + "_before_resume")
            before.mkdir()
            for filename in ("lingbotvla_cli.yaml", "run_info.txt", "config.yaml"):
                shutil.copy2(output / filename, before / filename)
        elif output.exists():
            raise FileExistsError(f"Preserving existing output: {output}")
        name = f"{run_name}_{phase}"
        log_path = state_dir / (name + ".log")
        env = os.environ.copy()
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy",
                    "WANDB_RUN_ID", "WANDB_RESUME", "WANDB_DISABLED", "WANDB_ANONYMOUS",
                    "LINGBOT_INFERENCE_CONFIG"):
            env.pop(key, None)
        env.update(WS=str(job), TRAIN_ENV=str(engine.TRAIN_ENV),
                   CONDA_ENVS_PATH=str(engine.TRAIN_ENV.parent), CONFIG=str(config_path),
                   CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", MICRO_BS="4", GLOBAL_BS="128",
                   NUM_WORKERS="8", PREFETCH_FACTOR="4", RUN_NAME=run_name,
                   MASTER_PORT="29581", PYTHONUNBUFFERED="1",
                   LD_PRELOAD=plan["ipc_so"], QL_TORCH_SHM_DIR=str(ipc),
                   WANDB_API_KEY=read(secret)["api_key"], WANDB_BASE_URL=logging["base_url"],
                   WANDB_ENTITY=logging["entity"], WANDB_PROJECT=logging["project"],
                   WANDB_MODE="online", WANDB_NAME=name, WANDB_GROUP=experiment,
                   WANDB_JOB_TYPE=phase, WANDB_TAGS="roi432,ee-guided,fixed-token-budget",
                   WANDB_DISABLE_CODE="true", WANDB_CONSOLE="off", WANDB_INIT_TIMEOUT="60",
                   QL_WANDB_SDK_PATH=logging["sdk_path"], WANDB_DIR=str(state_dir / "wandb"))
        env["QL_WANDB_RUN_METADATA"] = json.dumps(dict(
            execution_node="node1", code_commit=head, experiment="roi432_ee", phase=phase,
            input_resolution=[384, 384], num_cameras=3, expected_visual_tokens=432,
            micro_batch_size=4, global_batch_size=128, gradient_accumulation_steps=4,
            num_workers=8, prefetch_factor=4, pin_memory=True, seed=42,
            scheduler_total_steps=10000, resumed_from_step=resume_step,
            clean_norm_sha256=plan["clean_norm_sha256"], dataset_tasks=50, dataset_episodes=2500,
            roi_config_sha256=plan["roi_config_sha256"], output_dir=str(output),
            reference_config_sha256=plan["reference_config_sha256"],
            config_sha256=plan["config_sha256"], precision="enable_fp32=true",
            target_steps=target, checkpoint_every=2 if smoke else 1000))
        if resume:
            command = ["bash", str(repo / "xsy_utils/resolution_exp/resume.sh"),
                       str(output / f"checkpoints/global_step_{resume_step}")]
        else:
            command = ["bash", str(repo / "xsy_utils/resolution_exp/train.sh"), "384"]
        command += ["--train.max_steps", "10000", "--train.save_steps", "2" if smoke else "1000",
                    "--train.pause_after_steps", str(target if target < 10000 else 0),
                    "--train.enable_resume", "false", "--train.resume_dataloader_state", "true",
                    "--train.allow_dataloader_worker_change", "false",
                    "--data.num_workers", "8", "--data.prefetch_factor", "4",
                    "--train.seed", "42", "--train.enable_fp32", "true",
                    "--train.use_wandb", "true", "--train.wandb_project", logging["project"],
                    "--train.wandb_name", name]
        idle()
        with log_path.open("x") as log:
            child = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            engine.write_state(state, phase, child_pid=child.pid, target_steps=target,
                               resumed_from_step=resume_step, output=str(output), run_name=run_name,
                               log=str(log_path), scheduler_total_steps=10000,
                               micro_batch=4, global_batch=128, visual_tokens=432)
            code = child.wait()
        if code:
            raise RuntimeError(f"{phase} exited {code}; preserve claim, logs and checkpoints")
        text = log_path.read_text(errors="replace")
        if resume and f"Load distributed checkpoint from {output}/checkpoints/global_step_{resume_step} successfully!" not in text:
            raise RuntimeError("Missing explicit distributed resume confirmation")
        if re.search(r"(?:loss|gradnorm)[ =:]+[-+]?(?:nan|inf)\b", text, re.I):
            raise RuntimeError("Non-finite training metrics")
        engine.checkpoint_valid(output, target)
        engine.write_state(state, "phase_completed", child_pid=None)

    try:
        if args.mode == "smoke":
            train(True, False)
            train(True, True)
            result = "smoke_complete"
        else:
            train(False, False)
            engine.run_evaluation(state, experiment, 4, 5000, None)
            train(False, True)
            engine.run_evaluation(state, experiment, 4, 10000, None)
            result = "complete"
        engine.write_state(state, result, child_pid=None)
        owner.update(result=result, finished_utc=utc())
        atomic(claim / "owner.json", owner)
    except Exception as error:
        engine.write_state(state, "failed", error=str(error))
        owner.update(result="failed", error=str(error))
        atomic(claim / "owner.json", owner)
        raise


if __name__ == "__main__":
    main()
