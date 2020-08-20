from tensorboardX import SummaryWriter
from time import time
import shutil
import os


class EasySummaryWriter(SummaryWriter):

    def __init__(self, context, path='/tmp/tensorboard/', purge=True, **kwargs):
        print(f"tensorboard --logdir {path}")
        tensec = int(time() / 6)
        time_path = str(tensec - int(tensec / 100000.) * 100000)

        if purge:
            print(f"purge {path}")
            shutil.rmtree(path, ignore_errors=True)

        log_file = os.path.join(path, time_path, context)
        super().__init__(log_file, **kwargs)



