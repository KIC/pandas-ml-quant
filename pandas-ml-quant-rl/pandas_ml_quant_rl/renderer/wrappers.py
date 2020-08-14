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

    def render(self, mode=None):
        if not self.worker.is_alive():
            self.worker.start()

