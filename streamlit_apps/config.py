import os


class Config(object):

    app_home = os.path.dirname(os.path.abspath(__file__))

    model_directory = os.path.join(app_home, "models")

