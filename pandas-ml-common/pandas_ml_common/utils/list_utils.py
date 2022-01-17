from typing import List

import numpy as np


def as_empty_tuple(t) -> tuple:
    return (t if isinstance(t, tuple) else (t,)) if t is not None else tuple()


def none_as_empty_list(l):
    return (l if isinstance(l, List) else [l]) if l is not None else []


def get_first_or_tuple(e):
    if isinstance(e, (tuple, list)):
        if len(e) == 1:
            return e[0]
        else:
            return as_empty_tuple(e)
    else:
        return e


def make_same_length(item, reference_list):
    if not isinstance(item, List):
        item = [item]

    if len(item) == len(reference_list):
        return item
    elif len(item) == 1:
        return [item[0] for _ in reference_list]
    else:
        raise ValueError(f"incompatible length {len(item)}, {len(reference_list)}")


def safe_max(l):
    if l is None:
        return None
    l = [e for e in l if e is not None]
    return max(l) if len(l) > 0 else None


def safe_first(l):
    if l is None or not isinstance(l, List):
        return l

    return l[0] if len(l) > 0 else None


def fixsizelist(size, default=None):
    def fslist(elements):
        l = list(elements) + [default] * max(0, size - len(elements))
        return l[:size]

    return fslist