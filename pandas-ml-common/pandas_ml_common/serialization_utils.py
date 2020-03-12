import base64
import os
import io
from collections import Callable

import dill as pickle


def serialize(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

    print(f"saved model to: {os.path.abspath(filename)}")


def deserialize(filename, type):
    with open(filename, 'rb') as file:
        obj = pickle.load(file)

        if type is None:
            return obj

        if isinstance(obj, type):
            return obj
        else:
            raise ValueError("Deserialized pickle was not a Model!")


def plot_to_html_img(plotter: Callable):
    import matplotlib.pyplot as plt
    with io.BytesIO() as f:
        plotter()
        fig = plt.gcf()
        fig.savefig(f, format="png", bbox_inches='tight')
        image = base64.encodebytes(f.getvalue()).decode("utf-8")
        plt.close(fig)

        return f'data:image/png;base64, {image}'

