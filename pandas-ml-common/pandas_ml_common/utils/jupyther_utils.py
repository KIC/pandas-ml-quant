
def register_wirte_and_run_magic():
    from IPython.core.magic import register_cell_magic
    from IPython import get_ipython

    @register_cell_magic
    def write_and_run(line, cell):
        argz = line.split()
        file = argz[-1]
        mode = 'w'
        if len(argz) == 2 and argz[0] == '-a':
            mode = 'a'
        with open(file, mode) as f:
            f.write(cell)
        get_ipython().run_cell(cell)