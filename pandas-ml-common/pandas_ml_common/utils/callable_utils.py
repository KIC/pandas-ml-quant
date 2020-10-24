import inspect
import logging
from typing import Dict, Callable, Union, List, Tuple

_log = logging.getLogger(__name__)
_DEBUG = False


def call_silent(func, default=None):
    try:
        return func()
    except Exception as e:
        return default(e) if callable(default) else default


def kwpartial(func: Callable, **kwargs) -> Callable:
    """
    implements a partial function using kw arguments such that we can parse them back
    via the dependency injection: call_callable_dynamic_args

    :param func: a callable with keyword arguments only
    :param kwargs: key word arguments which get pre-assigned to the callable
    :return: a callable with named arguments except the once passed in and assigned by the kwargs argument
    """
    sa = suitable_kwargs(func, **kwargs)
    rest = inspect.getfullargspec(func).args
    rest = set(rest) - set(sa.keys())

    f = eval(f"lambda {','.join(rest)}: __func(**__suitable_kwargs, {','.join(['%s=%s' % (x, x) for x in rest])})",
             {"__func": func, "__suitable_kwargs": sa, **globals()})
    return f


def call_callable_dynamic_args(func, *args, **kwargs):
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


def merge_kwargs(*kwargs: Dict) -> Dict:
    dict = {}
    for d in kwargs:
        dict = {**dict, **d}

    return dict


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


class Signature:
    """1:1 copy of inspect.Signtature but without exceptions!"""
    __slots__ = ('_return_annotation', '_parameters')

    _parameter_cls = inspect.Parameter
    _bound_arguments_cls = inspect.BoundArguments

    empty = inspect._empty

    def __init__(self, parameters=None, *, return_annotation=inspect._empty, __validate_parameters__=True):
        if parameters is None:
            params = inspect.OrderedDict()
        else:
            if __validate_parameters__:
                params = inspect.OrderedDict()
                top_kind = inspect._POSITIONAL_ONLY
                kind_defaults = False

                for idx, param in enumerate(parameters):
                    kind = param.kind
                    name = param.name

                    if kind < top_kind:
                        msg = (
                            'wrong parameter order: {} parameter before {} '
                            'parameter'
                        )
                        msg = msg.format(inspect._get_paramkind_descr(top_kind), inspect._get_paramkind_descr(kind))
                        raise ValueError(msg)
                    elif kind > top_kind:
                        kind_defaults = False
                        top_kind = kind

                    if kind in (inspect._POSITIONAL_ONLY, inspect._POSITIONAL_OR_KEYWORD):
                        if param.default is inspect._empty:
                            if kind_defaults:
                                # No default for this parameter, but the
                                # previous parameter of the same kind had
                                # a default
                                msg = 'non-default argument follows default ' \
                                      'argument'
                                raise ValueError(msg)
                        else:
                            # There is a default for this parameter.
                            kind_defaults = True

                    if name in params:
                        msg = 'duplicate parameter name: {!r}'.format(name)
                        raise ValueError(msg)

                    params[name] = param
            else:
                params = inspect.OrderedDict(((param.name, param) for param in parameters))

        self._parameters = inspect.types.MappingProxyType(params)
        self._return_annotation = return_annotation


    @classmethod
    def from_callable(cls, obj, *, follow_wrapped=True):
        """Constructs Signature for the given callable object."""
        return inspect._signature_from_callable(obj, sigcls=cls, follow_wrapper_chains=follow_wrapped)

    @property
    def parameters(self):
        return self._parameters

    def replace(self, *, parameters=inspect._void, return_annotation=inspect._void):
        if parameters is inspect._void:
            parameters = self.parameters.values()

        if return_annotation is inspect._void:
            return_annotation = self._return_annotation

        return type(self)(parameters, return_annotation=return_annotation)

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
                            raise TypeError(msg) from None
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

    def bind(*args, **kwargs):
        return args[0]._bind(args[1:], kwargs)

    def bind_partial(*args, **kwargs):
        return args[0]._bind(args[1:], kwargs, partial=True)
