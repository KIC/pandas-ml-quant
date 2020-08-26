import torch as T


def grow_same_ndim(*tensor: T.tensor):
    max_dim = max([t.ndim for t in tensor])

    def grow(t):
        res = t
        while res.ndim < max_dim:
            res = res.unsqueeze(-1)

        return res

    return tuple([grow(t) for t in tensor])