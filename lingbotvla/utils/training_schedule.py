"""Separate an orderly checkpoint pause from the learning-rate horizon."""


def training_stop_step(max_steps, pause_after_steps=None, save_steps=None):
    if pause_after_steps is None or pause_after_steps == 0:
        return max_steps
    if max_steps is None or not 0 < pause_after_steps < max_steps:
        raise ValueError('pause_after_steps must be positive and below max_steps')
    if not save_steps or pause_after_steps % save_steps:
        raise ValueError('pause_after_steps must coincide with a regular checkpoint')
    return pause_after_steps


def capture_rng_state():
    import random
    import numpy as np
    import torch
    numpy_state = np.random.get_state()
    # DCP loads extra_state with torch.load(weights_only=True) on recent torch.
    # Keep numpy's uint32 array as plain values so the checkpoint stays loadable.
    numpy_state = (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:])
    return dict(torch=torch.get_rng_state(), python=random.getstate(), numpy=numpy_state,
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def restore_rng_state(state):
    import random
    import numpy as np
    import torch
    torch.set_rng_state(state['torch'])
    random.setstate(state['python'])
    numpy_state = state['numpy']
    np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'])
