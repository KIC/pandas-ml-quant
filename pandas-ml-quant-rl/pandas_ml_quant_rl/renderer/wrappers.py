import time
from queue import Empty
from multiprocessing import Process, Event, Queue
from .abstract_renderer import Renderer


def render_frame(data_q: Queue, finish_e: Event, renderer_provider):
    renderer = renderer_provider()

    while True:
        try:
            old_state, action, new_state, reward, done = data_q.get(timeout=0.1)
            renderer.plot(old_state, action, new_state, reward, done)
            renderer.render()
        except Empty:
            renderer.render()
            if finish_e.wait(0.1):
                break

    print("shut down online rendering !!!")


class OnlineRenderer(Renderer):

    def __init__(self, renderer_provider):
        super().__init__()
        self.data_q = Queue()
        self.finish_e = Event()
        self.worker = Process(target=render_frame, args=(self.data_q, self.finish_e, renderer_provider))
        self.startup = True

    def plot(self, old_state, action, new_state, reward, done):
        self.data_q.put_nowait((old_state, action, new_state, reward, done))
        if self.startup:
            time.sleep(1)
            self.startup = False

    def stop(self):
        self.finish_e.set()

    def render(self, mode=None, min_time_step=1.0):
        if not self.worker.is_alive():
            self.worker.start()


class MovieRenderer(Renderer):

    def __init__(self, renderer_provider):
        """
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as manimation

        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Movie Test', artist='Matplotlib',
                        comment='Movie support!')
        writer = FFMpegWriter(fps=15, metadata=metadata)

        fig = plt.figure()
        l, = plt.plot([], [], 'k-o')

        plt.xlim(-5, 5)
        plt.ylim(-5, 5)

        x0, y0 = 0, 0

        with writer.saving(fig, "writer_test.mp4", 100):
            for i in range(100):
                x0 += 0.1 * np.random.randn()
                y0 += 0.1 * np.random.randn()
                l.set_data(x0, y0)
                writer.grab_frame()
        :param renderer_provider:
        """
        pass
