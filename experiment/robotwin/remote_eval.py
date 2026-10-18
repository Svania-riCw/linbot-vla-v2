"""Validate RTC server capabilities and prepare resumable simulation-only runs."""
import argparse
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deploy.websocket_client_policy import WebsocketClientPolicy
from deploy.rtc_protocol import RTC_INFERENCE_TIME, RTC_TRAINING_TIME, normalize_rtc_mode
from experiment.robotwin.eval_checkpoint import file_digest, prepare_run


def validate_server(metadata, delay, replan_steps, rtc_mode=RTC_TRAINING_TIME):
    rtc_mode = normalize_rtc_mode(rtc_mode)
    if metadata.get('rtc_protocol') not in (1, 2):
        raise ValueError('Remote server must use a supported RTC protocol')
    if metadata.get('rtc_protocol') == 1 and rtc_mode == RTC_INFERENCE_TIME:
        raise ValueError('Inference-time RTC requires the protocol-2 server')
    horizon = int(metadata['action_horizon'])
    if rtc_mode == RTC_TRAINING_TIME:
        maximum = int(metadata['rtc_max_delay'])
        if not metadata.get('training_time_rtc'):
            raise ValueError('Training-time RTC requires an action-prefix-trained checkpoint')
        if not 0 <= delay < min(maximum, horizon):
            raise ValueError(f'Training-time RTC delay must be in [0, {min(maximum, horizon)-1}]')
    elif not 0 <= delay < horizon:
        raise ValueError(f'Inference-time RTC delay must be in [0, {horizon-1}]')
    server_modes = metadata.get('rtc_modes')
    if server_modes and rtc_mode not in server_modes:
        raise ValueError(f'Remote server does not support {rtc_mode!r}; it supports {server_modes}')
    if replan_steps < 1 or delay > replan_steps or replan_steps + delay > horizon:
        raise ValueError('RTC requires replan_steps >= delay and replan_steps + delay <= horizon')
    if not metadata.get('checkpoint_signature'):
        raise ValueError('Missing remote checkpoint signature')


def prepare(args, metadata):
    rtc_mode = normalize_rtc_mode(getattr(args, 'rtc_mode', RTC_TRAINING_TIME))
    validate_server(metadata, args.rtc_delay, args.rtc_replan_steps, rtc_mode)
    identity = dict(model=metadata.get('checkpoint', args.checkpoint_id),
                    remote_checkpoint_id=args.checkpoint_id,
                    remote_checkpoint_signature=metadata['checkpoint_signature'],
                    norm_stats_sha256=metadata.get('norm_stats_sha256'),
                    task_config=args.setting, task_config_sha256=file_digest(args.task_config_file),
                    instruction_type=args.instruction, seed=0, robo_name='robotwin',
                    denoiser=os.environ.get('ROBOTWIN_DENOISER', 'default'),
                    rtc=dict(mode=('training_time_prefix_conditioning_v1'
                                   if rtc_mode == RTC_TRAINING_TIME
                                   else 'inference_time_softmask_vjp_v1'),
                             delay=args.rtc_delay,
                             replan_steps=args.rtc_replan_steps))
    if rtc_mode == RTC_INFERENCE_TIME:
        identity['rtc'].update(guidance_beta=args.rtc_guidance_beta,
                               delay_buffer_size=4, scheduler='asap_adaptive_delay')
    prepare_run(args, identity)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for option in ('host','checkpoint-id','run-dir','setting','instruction','task-config-file'):
        parser.add_argument('--'+option,required=True)
    parser.add_argument('--port',required=True,type=int)
    parser.add_argument('--episodes',required=True,type=int)
    parser.add_argument('--tasks',required=True,nargs='+')
    parser.add_argument('--rtc-delay',type=int,default=6)
    parser.add_argument('--rtc-mode',choices=['training_time','inference_time','fixed_prefix','paper'],
                        default=RTC_TRAINING_TIME)
    parser.add_argument('--rtc-replan-steps',type=int,default=8)
    parser.add_argument('--rtc-guidance-beta',type=float,default=5.0)
    parser.add_argument('--timeout',type=float,default=180)
    args=parser.parse_args()
    try:
        client=WebsocketClientPolicy(args.host,args.port,request_timeout=args.timeout,startup_timeout=args.timeout)
        try:
            metadata=client.get_server_metadata()
        finally:
            client.close()
        prepare(args,metadata)
        print(f"Remote checkpoint: {metadata.get('checkpoint')}; signature: {metadata['checkpoint_signature']}")
    except (OSError, TimeoutError, ValueError, KeyError) as error:
        parser.exit(2,f'Error: {error}\n')


if __name__=='__main__':
    main()
