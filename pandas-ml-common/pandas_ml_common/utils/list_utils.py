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

