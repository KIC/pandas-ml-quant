import os
import shutil
from pathlib import Path

if __name__ == '__main__':
    eggs_home_dir = os.path.dirname(os.path.abspath(__file__))
    egg_dirs = [os.path.join(eggs_home_dir, dir) for dir in os.listdir(eggs_home_dir) if dir.endswith(".egg-info")]

    kernels_home_dir = os.path.join(Path.home(), ".local", "share", "jupyter", "kernels")
    kernel_dirs = [os.path.join(kernels_home_dir, dir) for dir in os.listdir(kernels_home_dir) if "tox" in dir]

    for dir in [*egg_dirs, *kernel_dirs]:
        print(f"clean up dir {dir}")

        try:
            shutil.rmtree(dir)
        except Exception as e:
            print("An exception occurred while removing *.egg-info")
            print(e)
