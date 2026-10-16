"""Check an explicitly exported, physical-unit NPZ; never infer data semantics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from action_contracts import ARM, GRIP, encode_joint_delta, decode_joint_delta


def check_arrays(data):
    state, actions = data['observed_state'], data['actions']
    delta = encode_joint_delta(actions, state)
    if actions.ndim != 3 or len(actions) == 0:
        raise ValueError("dataset must be nonempty [N,H,14]")
    pad = data['action_is_pad']
    episode, action_episode = data['episode_id'], data['action_episode_id']
    if pad.shape != actions.shape[:2] or pad.dtype != np.bool_:
        raise ValueError("action_is_pad must be bool [N,H]")
    if episode.shape != (len(actions),) or action_episode.shape != pad.shape:
        raise ValueError("episode ID shape mismatch")
    if np.any(np.all(pad, axis=1)):
        raise ValueError("all-padding chunk")
    if np.any(pad[:, :-1] & ~pad[:, 1:]):
        raise ValueError("padding must be a suffix")
    valid = ~pad
    if np.any((episode[:, None] != action_episode) & valid):
        raise ValueError("chunk crosses episode boundary")
    decoded = decode_joint_delta(delta, state)
    error = float(np.max(np.abs(decoded[valid] - actions[valid])))
    if error > 1e-5:
        raise ValueError(f"joint roundtrip exceeds 1e-5 rad: {error}")
    if not np.array_equal(delta[..., GRIP], actions[..., GRIP]):
        raise ValueError("gripper targets changed")
    absolute_q, delta_q = actions[..., ARM][valid], delta[..., ARM][valid]
    return {
        "status": "PASS_DATA_CONTRACT_ONLY", "chunks": len(actions),
        "valid_targets": int(valid.sum()), "pad_fraction": float(pad.mean()),
        "roundtrip_max_abs_error": error,
        "absolute_joint_std": absolute_q.std(axis=0).tolist(),
        "delta_joint_std": delta_q.std(axis=0).tolist(),
        "absolute_joint_abs_p95": np.quantile(np.abs(absolute_q), .95, axis=0).tolist(),
        "delta_joint_abs_p95": np.quantile(np.abs(delta_q), .95, axis=0).tolist(),
        "note": "Excludes pad for diagnostics only. No training mask or norm stats changed. Smaller std is not evidence of higher task success. Episode/time/unit correctness still depends on the exporter.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        result = check_arrays(data)
    result['input_sha256'] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result[k] for k in ('status', 'chunks', 'roundtrip_max_abs_error')}))


if __name__ == '__main__':
    main()
