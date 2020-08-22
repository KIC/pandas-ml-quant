import numpy as np


def discount_rewards(rewards, gamma=0.99) -> np.ndarray:
    # discount reward
    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])

    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()
