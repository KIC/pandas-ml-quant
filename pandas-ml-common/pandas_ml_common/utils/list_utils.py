from typing import List


def none_as_empty_list(l):
    return (l if isinstance(l, List) else [l]) if l is not None else []


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
