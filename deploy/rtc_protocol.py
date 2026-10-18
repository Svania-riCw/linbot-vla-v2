"""Validation for the single-flight RTC websocket protocol (no torch dependency)."""
import numpy as np

RTC_OFF = 'off'
RTC_TRAINING_TIME = 'training_time'
RTC_INFERENCE_TIME = 'inference_time'

# Wire names used before the two-paper split.  Keep them accepted so an old
# launcher can be upgraded independently from the server, while all new
# requests and manifests use the paper-aligned names above.
RTC_FIXED_PREFIX = 'async_fixed_prefix_v2'
RTC_PAPER = 'paper'
_RTC_MODE_ALIASES = {
    RTC_OFF: RTC_OFF,
    RTC_TRAINING_TIME: RTC_TRAINING_TIME,
    'fixed_prefix': RTC_TRAINING_TIME,
    RTC_FIXED_PREFIX: RTC_TRAINING_TIME,
    RTC_INFERENCE_TIME: RTC_INFERENCE_TIME,
    'paper': RTC_INFERENCE_TIME,
    RTC_PAPER: RTC_INFERENCE_TIME,
}


def normalize_rtc_mode(mode, default=RTC_TRAINING_TIME):
    """Map legacy mode names to the two paper-aligned runtime modes."""
    if mode is None:
        return default
    if not isinstance(mode, str):
        raise ValueError(f'Unsupported RTC mode: {mode!r}')
    try:
        return _RTC_MODE_ALIASES[mode]
    except KeyError as error:
        raise ValueError(f'Unsupported RTC mode: {mode!r}') from error


def validate_rtc_request(rtc, config, batch_size):
    if not isinstance(rtc, dict):
        raise ValueError('_rtc must be a mapping')
    mode = normalize_rtc_mode(rtc.get('mode'))
    if mode == RTC_OFF:
        raise ValueError('RTC request mode cannot be off; omit _rtc for synchronous inference')
    if mode == RTC_TRAINING_TIME and not getattr(config, 'training_time_rtc', False):
        raise ValueError('Training-time RTC requires an RTC-trained checkpoint')
    request_id = rtc.get('request_id')
    if not isinstance(request_id, str) or not request_id:
        raise ValueError('RTC request_id must be a nonempty string')
    delay = rtc.get('delay')
    horizon, dim = config.n_action_steps, config.max_action_dim
    if isinstance(delay, bool) or not isinstance(delay, int) or not 0 <= delay < horizon:
        raise ValueError(f'RTC delay must be an integer in [0, {horizon - 1}]')
    if mode == RTC_TRAINING_TIME:
        maximum = int(getattr(config, 'training_time_rtc_max_delay', 0))
        if delay >= maximum:
            raise ValueError(f'Training-time RTC delay must be in [0, {min(maximum, horizon) - 1}]')
    else:
        execution_horizon = rtc.get('execution_horizon')
        if (isinstance(execution_horizon, bool) or not isinstance(execution_horizon, int)
                or execution_horizon < 0 or execution_horizon > horizon):
            raise ValueError('Inference-time RTC execution_horizon must be an integer in [0, horizon]')
        if execution_horizon == 0 and delay == 0:
            pass  # Initial inference has no previous chunk to inpaint against.
        elif execution_horizon < 1 or delay > execution_horizon or execution_horizon + delay > horizon:
            raise ValueError('Inference-time RTC requires delay <= execution_horizon <= horizon - delay')
        beta = rtc.get('guidance_beta', 5.0)
        if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not np.isfinite(beta) or beta <= 0:
            raise ValueError('Inference-time RTC guidance_beta must be a finite positive number')
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
    selected_mode = getattr(policy, 'inference_rtc_mode', RTC_OFF)
    rtc_modes = []
    if policy.config.training_time_rtc:
        rtc_modes.append(RTC_TRAINING_TIME)
    if not getattr(policy, 'use_compile', True):
        rtc_modes.append(RTC_INFERENCE_TIME)
    if selected_mode != RTC_OFF:
        if selected_mode not in rtc_modes:
            raise ValueError(f'Server configuration cannot provide RTC mode {selected_mode!r}')
        rtc_modes = [selected_mode]
    mode = selected_mode
    return dict(rtc_protocol=2, rtc_mode=mode, rtc_modes=rtc_modes,
                training_time_rtc=bool(policy.config.training_time_rtc),
                rtc_max_delay=policy.config.training_time_rtc_max_delay,
                action_horizon=policy.config.n_action_steps, normalized_action_dim=policy.config.max_action_dim,
                checkpoint_signature=signature, checkpoint=str(root),
                norm_stats_sha256=identity['norm_stats_sha256'])
