import base64
import io
import os
import traceback
from collections import Callable

import dill as pickle


def serialize(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

    print(f"saved {type(obj)} to: {os.path.abspath(filename)}")


def deserialize(filename, type=None):
    with open(filename, 'rb') as file:
        obj = pickle.load(file)

        if type is None:
            return obj

        if isinstance(obj, type):
            return obj
        else:
            raise ValueError(f"Deserialized pickle was {type(obj)} but expected {type}!")


def plot_to_html_img(plotter: Callable, **kwargs):
    import matplotlib.pyplot as plt
    with io.BytesIO() as f:
        try:
            from pandas_ml_common.utils.callable_utils import call_callable_dynamic_args
            call_callable_dynamic_args(plotter, **kwargs)
            fig = plt.gcf()
            fig.savefig(f, format="png", bbox_inches='tight')
            image = base64.encodebytes(f.getvalue()).decode("utf-8")
            plt.close(fig)

            return f'data:image/png;base64, {image}'
        except TypeError:
            return traceback.print_exc()


def dict_to_str(d):
    if d is None:
        return ""
    else:
        from sortedcontainers import SortedDict
        return ",".join([f"{k}={v}" for k, v in SortedDict(d).items()])

