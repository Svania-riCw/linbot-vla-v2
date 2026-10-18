# RTC methods

This branch provides both methods described in the RTC papers:

- **Inference-time RTC** from [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/abs/2506.07339): no RTC retraining; asynchronous inpainting with a soft overlap mask and per-step VJP guidance. Use `--rtc_mode inference_time --use_compile False`. See [inference-time setup and evaluation](inference_time_rtc.md).
- **Training-time RTC** from [Training-Time Action Conditioning for Efficient Real-Time Chunking](https://arxiv.org/abs/2512.05964): simulate inference delay during training, condition on a clean ground-truth prefix, and train the postfix. At runtime it uses the fast hard-prefix sampler. See [training and checkpoint setup](training_time_rtc.md).

The canonical runtime modes are `inference_time` and `training_time`. The older
`paper` and `fixed_prefix` values remain accepted for compatibility.
