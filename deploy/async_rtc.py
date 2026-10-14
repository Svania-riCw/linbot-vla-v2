"""Single-flight RTC scheduling. Only the caller may observe/step the simulator."""
from concurrent.futures import ThreadPoolExecutor
import json
import time
import uuid

import numpy as np


def snapshot_observation(observation):
    """Detach camera/state buffers before the inference thread can access them."""
    return {key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in observation.items()}


class AsyncRTC:
    def __init__(self, model, replan_steps=8, delay=6, timeout=180.0, log=print):
        if replan_steps < 1 or delay < 0 or timeout <= 0:
            raise ValueError('Invalid RTC replan interval, delay, or timeout')
        self.model, self.replan_steps, self.delay = model, replan_steps, delay
        self.timeout, self.log = timeout, log
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rtc-infer')
        self.episode = uuid.uuid4().hex
        self.pending = None
        self.request_number = 0
        self.step = self.origin = self.cursor = self.executed = 0
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
        delay = 0 if initial else self.delay
        rtc = dict(request_id=request_id, delay=delay)
        if not initial:
            if self.cursor + delay > len(self.normalized):
                raise RuntimeError('Old plan has insufficient committed prefix')
            previous = np.zeros_like(self.normalized)
            previous[:delay] = self.normalized[self.cursor:self.cursor + delay]
            rtc['previous_actions'] = previous
        observation['_rtc'] = rtc
        origin = self.step
        start = time.monotonic()
        self.pending = (self.executor.submit(self.model.infer, observation), origin,
                        start, rtc)
        self.requests += 1
        self._log('request', request_id=request_id, delay=delay, cursor=self.cursor)

    def _install(self):
        future, origin, start, rtc = self.pending
        waiting = time.monotonic()
        if not future.done():
            self.stalls += 1
            self._log('wait', request_id=rtc['request_id'])
        response = future.result(timeout=max(0.001, self.timeout - (waiting - start)))
        waited = time.monotonic() - waiting
        self.wait_seconds += waited
        if response.get('_rtc_request_id') != rtc['request_id']:
            raise RuntimeError('Stale or unsupported RTC response request_id')
        raw = np.asarray(response['action'], dtype=np.float32)
        normalized = np.asarray(response['_normalized_actions'], dtype=np.float32)
        if (raw.ndim != 2 or normalized.ndim != 2 or len(raw) != len(normalized)
                or not np.isfinite(raw).all() or not np.isfinite(normalized).all()):
            raise RuntimeError('Invalid RTC action plan')
        if self.replan_steps + 2 * self.delay >= len(raw):
            raise ValueError('RTC requires replan_steps + 2 * delay < plan horizon')
        delay = rtc['delay']
        if delay:
            # Returned BF16 values are the exact normalized prefix stored by the model.
            if not np.allclose(normalized[:delay], rtc['previous_actions'][:delay], atol=1e-5, rtol=1e-5):
                raise RuntimeError('Server did not preserve normalized RTC prefix')
            if not np.allclose(raw[:delay], self.raw[self.cursor-delay:self.cursor], atol=1e-4, rtol=1e-4):
                raise RuntimeError('RTC prefix changed after action denormalization')
        if self.step - origin != delay:
            raise RuntimeError('RTC response is misaligned with executed target actions')
        self.raw, self.normalized = raw, normalized
        self.origin, self.cursor, self.executed = origin, delay, 0
        self.pending = None
        self._log('install', request_id=rtc['request_id'], skip=delay,
                  latency_s=time.monotonic()-start, wait_s=waited,
                  server_timing=response.get('server_timing'))

    def next_action(self, observe):
        if self.closed:
            raise RuntimeError('RTC scheduler is closed')
        if self.raw is None:
            self._submit(observe, initial=True)
            self._install()
        if self.pending is None and self.executed >= self.replan_steps:
            self._submit(observe)
        if self.pending is not None and self.step - self.pending[1] == self.delay:
            self._install()
        if self.cursor >= len(self.raw):
            raise RuntimeError('RTC action plan exhausted')
        action = self.raw[self.cursor].copy()
        self.cursor += 1
        self.step += 1
        self.executed += 1
        return action

    def close(self):
        if self.closed:
            return
        try:
            if self.pending is not None:
                future, _, start, _ = self.pending
                # Drain before the next episode resets the same websocket/policy.
                future.result(timeout=max(0.001, self.timeout - (time.monotonic()-start)))
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
