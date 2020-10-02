import os
import shutil


if __name__ == '__main__':
    egg_dirs = [dir for dir in os.listdir(os.path.dirname(os.path.abspath(__file__))) if dir.endswith(".egg-info")]
    print(f"clean up egg {egg_dirs}")

    try:
        for egg_dir in egg_dirs:
            shutil.rmtree(egg_dir)
    except Exception as e:
        print("An exception occurred while removing *.egg-info")
        print(e)
