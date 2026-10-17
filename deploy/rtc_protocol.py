"""Validation for the single-flight RTC websocket protocol (no torch dependency)."""
import numpy as np

RTC_FIXED_PREFIX = 'async_fixed_prefix_v2'
RTC_PAPER = 'paper'


def validate_rtc_request(rtc, config, batch_size):
    if not isinstance(rtc, dict):
        raise ValueError('_rtc must be a mapping')
    mode = rtc.get('mode', RTC_FIXED_PREFIX)
    if mode not in (RTC_FIXED_PREFIX, RTC_PAPER):
        raise ValueError(f'Unsupported RTC mode: {mode!r}')
    if mode == RTC_FIXED_PREFIX and not getattr(config, 'training_time_rtc', False):
        raise ValueError('Fixed-prefix RTC requires an RTC-trained checkpoint')
    request_id = rtc.get('request_id')
    if not isinstance(request_id, str) or not request_id:
        raise ValueError('RTC request_id must be a nonempty string')
    delay = rtc.get('delay')
    horizon, dim = config.n_action_steps, config.max_action_dim
    if isinstance(delay, bool) or not isinstance(delay, int) or not 0 <= delay < horizon:
        raise ValueError(f'RTC delay must be an integer in [0, {horizon - 1}]')
    if mode == RTC_FIXED_PREFIX:
        maximum = int(getattr(config, 'training_time_rtc_max_delay', 0))
        if delay >= maximum:
            raise ValueError(f'Fixed-prefix RTC delay must be in [0, {min(maximum, horizon) - 1}]')
    else:
        execution_horizon = rtc.get('execution_horizon')
        if (isinstance(execution_horizon, bool) or not isinstance(execution_horizon, int)
                or execution_horizon < 0 or execution_horizon > horizon):
            raise ValueError('Paper RTC execution_horizon must be an integer in [0, horizon]')
        if execution_horizon == 0 and delay == 0:
            pass  # Initial inference has no previous chunk to inpaint against.
        elif execution_horizon < 1 or delay > execution_horizon or execution_horizon + delay > horizon:
            raise ValueError('Paper RTC requires delay <= execution_horizon <= horizon - delay')
        beta = rtc.get('guidance_beta', 5.0)
        if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not np.isfinite(beta) or beta <= 0:
            raise ValueError('Paper RTC guidance_beta must be a finite positive number')
    previous = rtc.get('previous_actions')
    if previous is None:
        if delay:
            raise ValueError('Nonzero delay requires normalized previous_actions')
        return None, delay, request_id
    previous = np.asarray(previous, dtype=np.float32)
    if batch_size == 1 and previous.shape == (horizon, dim):
        previous = previous[None]
    if previous.shape != (batch_size, horizon, dim) or not np.isfinite(previous).all():
        raise ValueError(f'previous_actions must be finite normalized actions with shape {(batch_size, horizon, dim)}')
    return previous, delay, request_id


def server_metadata(policy, checkpoint):
    """Stable identity for resume checks; doesn't transmit model weights."""
    import hashlib
    import json
    import os
    from pathlib import Path
    root = Path(checkpoint).resolve()
    identity = dict(checkpoint=str(root), config=policy.config.to_dict(),
                    shards=[dict(name=p.name, size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                            for p in sorted(root.glob('*.safetensors'))])
    norm = Path(policy.robot_norm_path)
    identity['norm_stats_sha256'] = hashlib.sha256(norm.read_bytes()).hexdigest()
    config = Path(os.environ.get('LINGBOT_INFERENCE_CONFIG') or root.parent.parent.parent / 'lingbotvla_cli.yaml')
    identity['inference_config_sha256'] = hashlib.sha256(config.read_bytes()).hexdigest()
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return dict(rtc_protocol=1, training_time_rtc=bool(policy.config.training_time_rtc),
                rtc_max_delay=policy.config.training_time_rtc_max_delay,
                action_horizon=policy.config.n_action_steps, normalized_action_dim=policy.config.max_action_dim,
                checkpoint_signature=signature, checkpoint=str(root),
                norm_stats_sha256=identity['norm_stats_sha256'])
