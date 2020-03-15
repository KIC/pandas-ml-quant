import os

TEMLATE_DIR = os.path.dirname(os.path.abspath(__file__))
FIT_TEMPLATE = os.path.join(TEMLATE_DIR, "fit.py.html")


def SELF_TEMPLATE(file):
    return os.path.join(TEMLATE_DIR, f"{os.path.basename(file)}.html")