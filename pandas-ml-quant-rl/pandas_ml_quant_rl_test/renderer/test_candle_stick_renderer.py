from time import sleep
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np

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
        df = load_symbol("GLD").iloc[-200:]
        r = OnlineRenderer(lambda: CandleStickRenderer())
        r.render('online')

        for i in range(20):
            r.plot(df.iloc[[i-1]], 1, df.iloc[[i]], np.random.random(1) - 0.5, i >= 19)
            sleep(0.5)

        # close window after n seconds
        sleep(2)
        r.reset()



