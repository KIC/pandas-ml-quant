import logging

_log = logging.getLogger(__name__)


def reduce(values, reducer, default=None):
    if reducer == 'sum':
        return values.sum()
    elif reducer == 'mean':
        return values.mean()
    else:
        if reducer is not None:
            _log.warning(f"Unknown reduction {reducer}, don't reduce")

        return values if default is None else default
