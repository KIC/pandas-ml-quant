
def register_wirte_and_run_magic():
    from IPython.core.magic import register_cell_magic
    from IPython import get_ipython
    cut_off_magic = '## CUT'

    @register_cell_magic
    def write_and_run(line, cell):
        argz = line.split()
        file = argz[-1]
        mode = 'w'

        if len(argz) == 2 and argz[0] == '-a':
            mode = 'a'

        with open(file, mode) as f:
            f.write(cell.split(cut_off_magic)[0] if cut_off_magic in cell else cell)

        get_ipython().run_cell(cell)


def notebook_name():
    from notebook import notebookapp
    import urllib
    import json
    import os
    import ipykernel

    """
    Returns the absolute path of the Notebook or None if it cannot be determined
    NOTE: works only when the security is token-based or there is also no password
    """
    connection_file = os.path.basename(ipykernel.get_connection_file())
    kernel_id = connection_file.split('-', 1)[1].split('.')[0]

    for srv in notebookapp.list_running_servers():
        try:
            if srv['token'] == '' and not srv['password']:  # No token and no password, ahem...
                req = urllib.request.urlopen(srv['url'] + 'api/sessions')
            else:
                req = urllib.request.urlopen(srv['url'] + 'api/sessions?token=' + srv['token'])

            sessions = json.load(req)
            for sess in sessions:
                if sess['kernel']['id'] == kernel_id:
                    return os.path.basename(sess['notebook']['path'])
        except:
            pass  # There may be stale entries in the runtime directory

    return None

