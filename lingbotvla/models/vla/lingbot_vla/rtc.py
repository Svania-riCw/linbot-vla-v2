"""Backward-compatible RTC helper imports.

New code should import ``training_time_rtc`` for arXiv:2512.05964 helpers and
``inference_time_rtc`` for arXiv:2506.07339 helpers.  This module remains as a
small compatibility shim for downstream users of the old import path.
"""

from .training_time_rtc import (
    align_previous_action_chunk,
    apply_rtc_prefix,
    combine_rtc_loss_mask,
    make_rtc_flow,
    rtc_prefix_mask,
    rtc_time_for_prefix,
    sample_rtc_delay,
)
from .inference_time_rtc import rtc_soft_mask

__all__ = [
    "align_previous_action_chunk",
    "apply_rtc_prefix",
    "combine_rtc_loss_mask",
    "make_rtc_flow",
    "rtc_prefix_mask",
    "rtc_soft_mask",
    "rtc_time_for_prefix",
    "sample_rtc_delay",
]
