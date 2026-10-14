"""Validation for the single-flight RTC websocket protocol (no torch dependency)."""
import numpy as np


def validate_rtc_request(rtc, config, batch_size):
    if not isinstance(rtc, dict):
        raise ValueError('_rtc must be a mapping')
    if not getattr(config, 'training_time_rtc', False):
        raise ValueError('RTC inference requires an RTC-trained checkpoint')
    request_id = rtc.get('request_id')
    if not isinstance(request_id, str) or not request_id:
        raise ValueError('RTC request_id must be a nonempty string')
    delay = rtc.get('delay')
    maximum = int(getattr(config, 'training_time_rtc_max_delay', 0))
    horizon, dim = config.n_action_steps, config.max_action_dim
    if isinstance(delay, bool) or not isinstance(delay, int) or not 0 <= delay < min(maximum, horizon):
        raise ValueError(f'RTC delay must be an integer in [0, {min(maximum, horizon) - 1}]')
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
