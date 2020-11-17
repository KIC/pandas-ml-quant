import matplotlib.pyplot as plt


class PlotContext(object):

    def __init__(self, df, width: int = 20, main_height: int = 11, tail: int = 5 * 52, backend='notebook'):
        self.old_backend = plt.get_backend()
        self.width = width
        self.main_height = main_height
        self.tail = tail
        self.target_backend = backend

    def __enter__(self):
        self.old_backend = plt.get_backend()
        print(self.old_backend)
        # bring back plotting:
        # def subplots(self, rows=2, figsize=(25, 10)):
        #     import matplotlib.pyplot as plt
        #     import matplotlib.dates as mdates
        #
        #     _, axes = plt.subplots(rows, 1,
        #                         sharex=True,
        #                         gridspec_kw={"height_ratios": [3, *([1] * (rows - 1))]},
        #                         figsize=figsize)
        #
        #     for ax in axes if isinstance(axes, Iterable) else [axes]:
        #         ax.xaxis_date()
        #         ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
        #
        #     return axes
        #
        # def plot(self, rows=2, cols=1, figsize=(18, 10), main_height_ratio=4):
        #     pass # return TaPlot(self.df, figsize, rows, cols, main_height_ratio)

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(f'exc_type: {exc_type}')
            print(f'exc_value: {exc_value}')
            print(f'exc_traceback: {exc_traceback}')

            plt.switch_backend(self.old_backend)
            # eventually return True if all excetions are handled

    def __str__(self):
        return f'{self.width}/{self.main_height} @ {self.target_backend}'
