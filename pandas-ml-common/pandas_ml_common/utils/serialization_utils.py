import base64
import io
import os
import traceback

import dill as pickle


def serialize(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

    print(f"saved {type(obj)} to: {os.path.abspath(filename)}")


def serializeb(obj):
    return pickle.dumps(obj)


def deserialize(filename, type=None):
    with open(filename, 'rb') as file:
        obj = pickle.load(file)

        if type is None:
            return obj
        elif isinstance(obj, type):
            return obj
        else:
            raise ValueError(f"Deserialized pickle was {type(obj)} but expected {type}!")


def deserializeb(bytes, type=None):
    obj = pickle.loads(bytes)

    if type is None:
        return obj
    elif isinstance(obj, type):
        return obj
    else:
        raise ValueError(f"Deserialized pickle was {type(obj)} but expected {type}!")


def plot_to_html_img(plotter, **kwargs):
    import matplotlib.pyplot as plt
    from pandas_ml_common.utils.callable_utils import call_callable_dynamic_args

    if callable(plotter):
        ret_fig = call_callable_dynamic_args(plotter, **kwargs)
        fig = ret_fig if isinstance(ret_fig, plt.Figure) else plt.gcf()
    else:
        fig = plotter

    with io.BytesIO() as f:
        try:
            fig.savefig(f, format="png", bbox_inches='tight')
            image = base64.encodebytes(f.getvalue()).decode("utf-8")

            return f'data:image/png;base64, {image}'
        except TypeError:
            return traceback.print_exc()
        finally:
            plt.close(fig)


def dict_to_str(d):
    if d is None:
        return ""
    else:
        from sortedcontainers import SortedDict
        return ",".join([f"{k}={v}" for k, v in SortedDict(d).items()])

