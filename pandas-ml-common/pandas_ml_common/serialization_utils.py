import os

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
