"""Single-flight RTC scheduling. Only the caller may observe/step the simulator."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from collections import deque
import json
import time
import uuid

import numpy as np

from .rtc_protocol import (
    RTC_FIXED_PREFIX,
    RTC_INFERENCE_TIME,
    RTC_PAPER,
    RTC_TRAINING_TIME,
    normalize_rtc_mode,
)


def snapshot_observation(observation):
    """Detach camera/state buffers before the inference thread can access them."""
    return {key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in observation.items()}


class AsyncRTC:
    """Overlap requests while counting the committed prefix in the interval.

    For example, ``replan_steps=8`` and ``delay=6`` starts each later request
    eight target actions after the previous one: six old-plan actions while the
    request runs, followed by two actions from the installed plan.
    """

    def __init__(self, model, replan_steps=8, delay=6, timeout=180.0, log=print,
                 mode=RTC_TRAINING_TIME, guidance_beta=5.0):
        if replan_steps < 1 or delay < 0 or timeout <= 0:
            raise ValueError('Invalid RTC replan interval, delay, or timeout')
        if delay > replan_steps:
            raise ValueError('RTC delay cannot exceed the interval between inference requests')
        mode = normalize_rtc_mode(mode)
        if mode == RTC_INFERENCE_TIME and (guidance_beta <= 0 or not np.isfinite(guidance_beta)):
            raise ValueError('Inference-time RTC guidance_beta must be finite and positive')
        self.model, self.replan_steps, self.delay = model, replan_steps, delay
        self.mode, self.guidance_beta = mode, float(guidance_beta)
        self.delay_history = deque([delay], maxlen=4)
        self.timeout, self.log = timeout, log
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rtc-infer')
        self.episode = uuid.uuid4().hex
        self.pending = None
        self.request_number = 0
        self.step = self.origin = self.cursor = self.steps_since_request = 0
        self.raw = self.normalized = None
        self.requests = self.stalls = 0
        self.wait_seconds = 0.0
        self.closed = False

    def _log(self, event, **fields):
        self.log('[rtc] ' + json.dumps(dict(event=event, step=self.step, **fields)), flush=True)

    def _submit(self, observe, initial=False):
        observation = snapshot_observation(observe())
        self.request_number += 1
        request_id = f'{self.episode}:{self.request_number}'
        delay = 0 if initial else (max(self.delay_history) if self.mode == RTC_INFERENCE_TIME else self.delay)
        wire_mode = self.mode
        metadata = getattr(self.model, '_server_metadata', {}) or {}
        if int(metadata.get('rtc_protocol', 2)) < 2:
            wire_mode = RTC_FIXED_PREFIX if self.mode == RTC_TRAINING_TIME else RTC_PAPER
        rtc = dict(request_id=request_id, delay=delay, mode=wire_mode)
        if self.mode == RTC_INFERENCE_TIME:
            rtc.update(mode=wire_mode, execution_horizon=0 if initial else self.cursor,
                       guidance_beta=self.guidance_beta)
        if not initial:
            if self.cursor + delay > len(self.normalized):
                raise RuntimeError('Old plan has insufficient committed prefix')
            if self.mode == RTC_INFERENCE_TIME and (delay > self.cursor or self.cursor + delay > len(self.normalized)):
                raise RuntimeError('Inference-time RTC needs an execution horizon at least as long as its delay estimate')
            previous = np.zeros_like(self.normalized)
            if self.mode == RTC_INFERENCE_TIME:
                # At request start the current cursor is the execution horizon
                # into the old chunk.  Shift the complete remaining chunk so
                # the server can apply the paper's soft overlap mask.
                remaining = self.normalized[self.cursor:]
                previous[:len(remaining)] = remaining
            else:
                previous[:delay] = self.normalized[self.cursor:self.cursor + delay]
            rtc['previous_actions'] = previous
        observation['_rtc'] = rtc
        origin = self.step
        start = time.monotonic()
        self.pending = (self.executor.submit(self.model.infer, observation), origin,
                        start, rtc)
        # The request interval includes target actions executed while inference
        # is pending, including the committed prefix.
        self.steps_since_request = 0
        self.requests += 1
        self._log('request', request_id=request_id, delay=delay, cursor=self.cursor)

    def _await_pending(self):
        future, _, start, rtc = self.pending
        try:
            return future.result(timeout=max(0.001, self.timeout - (time.monotonic() - start)))
        except FutureTimeoutError as error:
            # concurrent.futures.TimeoutError is distinct on simulator Python 3.10.
            raise TimeoutError(f"RTC request timed out: {rtc['request_id']}") from error

    def _install(self):
        future, origin, start, rtc = self.pending
        waiting = time.monotonic()
        if not future.done():
            self.stalls += 1
            self._log('wait', request_id=rtc['request_id'])
        response = self._await_pending()
        waited = time.monotonic() - waiting
        self.wait_seconds += waited
        if response.get('_rtc_request_id') != rtc['request_id']:
            raise RuntimeError('Stale or unsupported RTC response request_id')
        raw = np.asarray(response['action'], dtype=np.float32)
        normalized = np.asarray(response['_normalized_actions'], dtype=np.float32)
        if (raw.ndim != 2 or normalized.ndim != 2 or len(raw) != len(normalized)
                or not np.isfinite(raw).all() or not np.isfinite(normalized).all()):
            raise RuntimeError('Invalid RTC action plan')
        if self.mode == RTC_TRAINING_TIME and self.replan_steps + self.delay > len(raw):
            raise ValueError('RTC requires replan_steps + delay <= plan horizon')
        actual_delay = self.step - origin
        if actual_delay >= len(raw):
            raise ValueError('RTC inference delay exhausted the action-plan horizon')
        delay = rtc['delay']
        if delay and self.mode == RTC_TRAINING_TIME:
            # Returned BF16 values are the exact normalized prefix stored by the model.
            if not np.allclose(normalized[:delay], rtc['previous_actions'][:delay], atol=1e-5, rtol=1e-5):
                raise RuntimeError('Server did not preserve normalized RTC prefix')
            if not np.allclose(raw[:delay], self.raw[self.cursor-delay:self.cursor], atol=1e-4, rtol=1e-4):
                raise RuntimeError('RTC prefix changed after action denormalization')
        if self.mode == RTC_TRAINING_TIME and actual_delay != delay:
            raise RuntimeError('Training-time RTC response is misaligned with executed target actions')
        self.raw, self.normalized = raw, normalized
        self.origin = origin
        self.cursor = actual_delay if self.mode == RTC_INFERENCE_TIME else delay
        if self.mode == RTC_INFERENCE_TIME and origin > 0:
            self.delay_history.append(actual_delay)
        self.pending = None
        self._log('install', request_id=rtc['request_id'], skip=self.cursor,
                  predicted_delay=delay if self.mode == RTC_INFERENCE_TIME else None,
                  latency_s=time.monotonic()-start, wait_s=waited,
                  server_timing=response.get('server_timing'))

    def next_action(self, observe):
        if self.closed:
            raise RuntimeError('RTC scheduler is closed')
        if self.raw is None:
            self._submit(observe, initial=True)
            self._install()
        next_request_threshold = self.replan_steps
        if self.mode == RTC_INFERENCE_TIME:
            next_request_threshold = max(next_request_threshold, max(self.delay_history))
        if self.pending is None and self.steps_since_request >= next_request_threshold:
            self._submit(observe)
        if self.pending is not None:
            should_install = (
                self.pending[0].done() if self.mode == RTC_INFERENCE_TIME
                else self.step - self.pending[1] == self.delay
            )
            if should_install:
                self._install()
            # If delay equals the request interval, the next request is due
            # immediately after installing this plan, before executing it.
            if self.pending is None and self.steps_since_request >= next_request_threshold:
                self._submit(observe)
        if self.cursor >= len(self.raw):
            if self.pending is not None and self.mode == RTC_INFERENCE_TIME:
                self._install()
            else:
                raise RuntimeError('RTC action plan exhausted')
        action = self.raw[self.cursor].copy()
        self.cursor += 1
        self.step += 1
        self.steps_since_request += 1
        return action

    def close(self):
        if self.closed:
            return
        try:
            if self.pending is not None:
                future, _, start, _ = self.pending
                # Drain before the next episode resets the same websocket/policy.
                self._await_pending()
                self.pending = None
        except BaseException:
            close = getattr(self.model, 'close', None)
            if close:
                close()  # Unblock a recv on errors/timeouts before joining the worker.
            raise
        finally:
            self.closed = True
            self.executor.shutdown(wait=True, cancel_futures=True)
            self._log('summary', requests=self.requests, stalls=self.stalls,
                      wait_s=self.wait_seconds)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
