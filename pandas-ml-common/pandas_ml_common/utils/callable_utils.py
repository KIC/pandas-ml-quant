import inspect
from typing import Dict


def call_callable_dynamic_args(func, *args, **kwargs):
    if not callable(func):
        raise ValueError(f"function {func} is not callable")

    spec = inspect.getfullargspec(func)
    call_args = []

    for i in range(len(spec.args)):
        if i < len(args):
            call_args.append(args[i])
        elif spec.args[i] in kwargs:
            call_args.append(kwargs[spec.args[i]])
            del kwargs[spec.args[i]]

    # inject eventually still missing var args
    if spec.varargs and len(args) > len(spec.args) and len(args) > len(call_args):
        call_args += args[len(call_args):]

    try:
        if spec.varkw:
            # inject the rest of kwargs if we have some left overs
            return func(*call_args, **kwargs)
        else:
            return func(*call_args)
    except Exception as e:
        raise RuntimeError(e, f"error while calling {func} whith arguments {call_args}")


def suitable_kwargs(func, **kwargs):
    suitable_args = inspect.getfullargspec(func).args
    return {arg: kwargs[arg] for arg in kwargs.keys() if arg in suitable_args}


def call_if_not_none(obj, method, *args, **kwargs):
    if obj is None:
        return None
    else:
        attr = getattr(obj, method)
        return attr(*args, ** kwargs) if callable(attr) else attr


def merge_kwargs(*kwargs: Dict) -> Dict:
    dict = {}
    for d in kwargs:
        dict = {**dict, **d}

    return dict
