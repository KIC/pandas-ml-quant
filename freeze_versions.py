import subprocess
import sys


def freeze_versions(requirements):
    needed_packages = [r.replace('\r', '').replace('\n', '') for r in open(requirements, 'r').readlines() if not r.startswith("#")]
    # print(needed_packages)

    sp = subprocess.Popen("pip freeze", shell=True, stdout=subprocess.PIPE)
    current_versions = [r.decode('UTF-8').replace('\r', '').replace('\n', '') for r in sp.stdout.readlines()]
    # print(current_versions)

    frozen_packages = [cv for cv in current_versions if cv.split('==')[0] in needed_packages]
    # print(frozen_packages)

    assert len(needed_packages) == len(frozen_packages), \
           f'Not all libraries could be found/frozen!\n{needed_packages}\n{frozen_packages}'

    return frozen_packages


def get_frozen_filename(filename):
    parts = filename.split('.')
    parts = parts[:-1] + ['frozen'] + parts[-1:]
    return '.'.join(parts)


if __name__ == '__main__':
    for requirements_file in sys.argv[1:]:
        frozen_dependencies = freeze_versions(requirements_file)
        frozen_file_name = get_frozen_filename(requirements_file)
        open(frozen_file_name, 'w').write('\n'.join(frozen_dependencies))
