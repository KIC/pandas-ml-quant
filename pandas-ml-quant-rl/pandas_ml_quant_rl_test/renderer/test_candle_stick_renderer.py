from time import sleep
from unittest import TestCase

import matplotlib.pyplot as plt

from pandas_ml_quant_rl.renderer import CandleStickRenderer
from pandas_ml_quant_rl.renderer.abstract_renderer import OnlineRenderer
from pandas_ml_quant_rl_test.config import load_symbol


DEV = True


def keep_plot():
    if DEV:
        # keep plot open
        plt.ioff()
        plt.show()


class TestCandleStickRenderer(TestCase):

    def test_render(self):
        df = load_symbol("SPY")
        r = OnlineRenderer(lambda: CandleStickRenderer())
        r.render('online')

        for i in range(10):
            r.plot(df.iloc[[i-1]], 1, df.iloc[[i]], 0, False)
            sleep(0.5)

        r.wait()



