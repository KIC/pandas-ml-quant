import inspect
import logging
from typing import Dict, Callable, Union, List, Tuple, Iterable

_log = logging.getLogger(__name__)
_DEBUG = False


def call_silent(func, default=None, log=False, log_ctx=None):
    try:
        return func()
    except Exception as e:
        if log:
            _log.warning(f"failed {func} call: {log_ctx}: {e}")
        return default(e) if callable(default) else default


def call_callable_dynamic_args(func, *args, **kwargs):
    if isinstance(func, Iterable):
        return [call_callable_dynamic_args(f, *args, **kwargs) for f in func]

    if not callable(func):
        if func is None:
            return None
        else:
            raise ValueError(f"function {func} is not callable")

    callable_args = Signature.from_callable(func).bind(*args, **kwargs)

    try:
        return func(*callable_args.args, **callable_args.kwargs)
    except StopIteration as s:
        raise s
    except KeyError as ke:
        raise ke
    except Exception as e:
        try:
            source = inspect.getsource(func)
        except OSError:
            try:
                from dill.source import getsource
                source = getsource(func)
            except Exception:
                source = "eval"

        raise RuntimeError(e, f"error while calling {func}({inspect.getfullargspec(func)})\n{source}\nwith arguments:\n{callable_args}, {kwargs}")


def suitable_kwargs(func, *args, **kwargs):
    _kwargs = {}

    for arg in args:
        _kwargs = {**_kwargs, **arg}

    _kwargs = {**_kwargs, **kwargs}
    suitable_args = inspect.getfullargspec(func).args
    return {arg: _kwargs[arg] for arg in _kwargs.keys() if arg in suitable_args}


def exec_if_not_none(func, obj):
    return func(obj) if obj is not None else None


def call_if_not_none(obj, method, *args, **kwargs):
    if obj is None:
        return None
    else:
        attr = getattr(obj, method)
        return attr(*args, ** kwargs) if callable(attr) else attr


def merge_kwargs(*args_kwargs: Dict, **kwargs) -> Dict:
    dict = {}
    for d in args_kwargs:
        dict = {**dict, **d}

    return {**dict, **kwargs}


def flatten_nested_list(l: Union[List, Tuple], func: callable) -> List:
    if isinstance(l, (List, Tuple)):
        res = []

        for i in l:
            r = flatten_nested_list(i, func)
            if isinstance(r, (List, Tuple)):
                res.extend(r)
            else:
                res.append(r)

        return res
    else:
        return [func(l)]


class Signature(inspect.Signature):
    """1:1 copy of inspect.Signtature but without exceptions!"""

    def __init__(self, parameters=None, *, return_annotation=inspect._empty, __validate_parameters__=True):
        super().__init__(parameters=parameters, return_annotation=return_annotation)

    @classmethod
    def from_callable(cls, obj, *, follow_wrapped=True):
        """Constructs Signature for the given callable object."""
        return inspect._signature_from_callable(obj, sigcls=cls, follow_wrapper_chains=follow_wrapped)

    def _bind(self, args, kwargs, *, partial=False):
        """Private method. Don't use directly."""

        arguments = inspect.OrderedDict()

        parameters = iter(self.parameters.values())
        parameters_ex = ()
        arg_vals = iter(args)

        while True:
            # Let's iterate through the positional arguments and corresponding
            # parameters
            try:
                arg_val = next(arg_vals)
            except StopIteration:
                # No more positional arguments
                try:
                    param = next(parameters)
                except StopIteration:
                    # No more parameters. That's it. Just need to check that
                    # we have no `kwargs` after this while loop
                    break
                else:
                    if param.kind == inspect._VAR_POSITIONAL:
                        # That's OK, just empty *args.  Let's start parsing
                        # kwargs
                        break
                    elif param.name in kwargs:
                        if param.kind == inspect._POSITIONAL_ONLY:
                            msg = '{arg!r} parameter is positional only, ' \
                                  'but was passed as a keyword'
                            msg = msg.format(arg=param.name)
                            raise TypeError(msg) from None
                        parameters_ex = (param,)
                        break
                    elif (param.kind == inspect._VAR_KEYWORD or param.default is not inspect._empty):
                        # That's fine too - we have a default value for this
                        # parameter.  So, lets start parsing `kwargs`, starting
                        # with the current parameter
                        parameters_ex = (param,)
                        break
                    else:
                        # No default, not VAR_KEYWORD, not VAR_POSITIONAL,
                        # not in `kwargs`
                        if partial:
                            parameters_ex = (param,)
                            break
                        else:
                            msg = 'missing a required argument: {arg!r}'
                            msg = msg.format(arg=param.name)
                            # raise TypeError(msg) from None
            else:
                # We have a positional argument to process
                try:
                    param = next(parameters)
                except StopIteration:
                    continue
                else:
                    if param.kind in (inspect._VAR_KEYWORD, inspect._KEYWORD_ONLY):
                        # Looks like we have no parameter for this positional
                        # argument
                        raise TypeError('too many positional arguments') from None

                    if param.kind == inspect._VAR_POSITIONAL:
                        # We have an '*args'-like argument, let's fill it with
                        # all positional arguments we have left and move on to
                        # the next phase
                        values = [arg_val]
                        values.extend(arg_vals)
                        arguments[param.name] = tuple(values)
                        break

                    arguments[param.name] = arg_val

        # Now, we iterate through the remaining parameters to process
        # keyword arguments
        kwargs_param = None
        for param in inspect.itertools.chain(parameters_ex, parameters):
            if param.kind == inspect._VAR_KEYWORD:
                # Memorize that we have a '**kwargs'-like parameter
                kwargs_param = param
                continue

            if param.kind == inspect._VAR_POSITIONAL:
                # Named arguments don't refer to '*args'-like parameters.
                # We only arrive here if the positional arguments ended
                # before reaching the last parameter before *args.
                continue

            param_name = param.name
            try:
                arg_val = kwargs.pop(param_name)
            except KeyError:
                # We have no value for this parameter.  It's fine though,
                # if it has a default value, or it is an '*args'-like
                # parameter, left alone by the processing of positional
                # arguments.
                if (not partial and param.kind != inspect._VAR_POSITIONAL and param.default is inspect._empty):
                    raise TypeError('missing a required argument: {arg!r}'.format(arg=param_name)) from None

            else:
                if param.kind == inspect._POSITIONAL_ONLY:
                    # This should never happen in case of a properly built
                    # Signature object (but let's have this check here
                    # to ensure correct behaviour just in case)
                    raise TypeError('{arg!r} parameter is positional only, '
                                    'but was passed as a keyword'. \
                                    format(arg=param.name))

                arguments[param_name] = arg_val

        if kwargs:
            if kwargs_param is not None:
                # Process our '**kwargs'-like parameter
                arguments[kwargs_param.name] = {k: v for k, v in kwargs.items() if k not in arguments}

        return self._bound_arguments_cls(self, arguments)
