"""Display evaluation progress from logs, without connecting to the simulator."""
import argparse
import json
import re
import sys
import time
from pathlib import Path

ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
STAGES = {
    'waiting_server': '等待推理服务/模型加载',
    'expert_check': '专家轨迹检查',
    'scene_setup': '初始化评测场景',
    'recording_setup': '初始化录像',
    'model_reset': '重置模型',
    'observation': '获取相机观测',
    'inference': '模型推理',
    'actions': '执行动作/录制视频',
    'video_finalize': '完成视频编码',
    'episode_done': 'episode 完成',
}


def read_progress(log, episodes):
    path = Path(log)
    try:
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - 131072))
            content = ANSI.sub('', stream.read().decode('utf-8', errors='replace'))
        age = max(0, int(time.time() - path.stat().st_mtime))
    except FileNotFoundError:
        content, age = '', 0
    # A retry starts a fresh evaluation; ignore the failed attempt's progress.
    content = re.split(r'\[attempt \d+/\d+\]', content)[-1]
    state = dict(stage='starting', episode=1, episodes=episodes,
                 completed=0, successes=0, step=0, limit=0)
    markers = list(re.finditer(r'\[eval-progress\] (\{[^\r\n]*\})', content))
    if markers:
        try:
            state.update(json.loads(markers[-1].group(1)))
            step_content = content[markers[-1].end():]
        except json.JSONDecodeError:
            step_content = content
    else:
        step_content = content
        rates = re.findall(r'Success rate: (\d+)/(\d+)', content)
        if rates:
            successes, completed = map(int, rates[-1])
            state.update(successes=successes, completed=completed,
                         episode=min(completed + 1, episodes))
            step_content = content[content.rfind('Success rate:'):]
            state['stage'] = 'episode_done' if completed >= episodes else 'starting'
    steps = re.findall(r'step:\s*(\d+)\s*/\s*(\d+)', step_content)
    if steps:
        state['step'], state['limit'] = map(int, steps[-1])
        state['stage'] = 'actions'
    if 'Traceback (most recent call last)' in step_content:
        state['stage'] = 'error'
    state['age'] = age
    return state


def format_progress(task, state, elapsed):
    step, limit = state['step'], state['limit']
    fraction = min(1, max(0, step / limit)) if limit else 0
    filled = int(fraction * 20)
    bar = '#' * filled + '-' * (20 - filled)
    actions = f'{step}/{limit} {fraction:.0%}' if limit else '等待动作'
    stage = STAGES.get(state['stage'], {'starting': '启动/初始化', 'error': '异常，检查日志'}.get(state['stage'], state['stage']))
    return (f"[progress] {task} | episode {state['episode']}/{state['episodes']} "
            f"| [{bar}] {actions} | {stage} | 已完成 {state['completed']}/{state['episodes']} "
            f"成功 {state['successes']} | 已用 {elapsed}s | 日志 {state['age']}s 前更新")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--elapsed', type=int, default=0)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    start = time.monotonic()
    try:
        while True:
            state = read_progress(args.log, args.episodes)
            text = format_progress(args.task, state, args.elapsed + int(time.monotonic() - start))
            inline = args.watch and sys.stdout.isatty()
            print(('\r\033[2K' if inline else '') + text, end='' if inline else '\n', flush=True)
            if not args.watch or state['completed'] >= state['episodes'] or state['stage'] == 'error':
                if inline:
                    print()
                break
            time.sleep(5)
    except KeyboardInterrupt:
        print()


if __name__ == '__main__':
    main()
