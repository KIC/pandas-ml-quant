import multiprocessing as mp
import random
import time
from queue import Empty

import matplotlib.pyplot as plt
import numpy


def worker(q):
    #plt.ion()
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ln, = ax.plot([], [])
    fig.canvas.draw()   # draw and show it
    plt.show(block=False)
    x = 0

    while True:
        try:
            obj = q.get(timeout = 0.8)
            n = obj + 0
            x += 1
            print("sub : got:", n)

            ax.bar([x], [n])

        except Empty as e:
            ax.autoscale_view(tight=True, scalex=True, scaley=False)
            fig.canvas.draw()
            plt.pause(0.05)
            continue


if __name__ == '__main__':
    queue = mp.Queue()
    p = mp.Process(target=worker, args=(queue,))
    p.start()

    while True:
        n = random.random() * 5
        queue.put(n)
        time.sleep(1.0)

