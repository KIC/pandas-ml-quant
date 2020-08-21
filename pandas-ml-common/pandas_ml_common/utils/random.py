import contextlib
import numpy as np


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def normalize_probabilities(probs):
    if probs is not None:
        if not isinstance(probs, np.ndarray):
            probs = np.array(probs)
        return probs / probs.sum()
    else:
        return None

