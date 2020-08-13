import multiprocessing
import time
from queue import Empty


def render_frame(q, reset_event, renderer_provider):
    renderer = renderer_provider()

    done = False
    while not done:
        try:
            old_state, action, new_state, reward, done = q.get(timeout=0.2)
            renderer.plot(old_state, action, new_state, reward, done)
        except Empty:
            renderer.render()

    reset_event.wait()
    renderer.reset()
    
    print("done !!!")


class Renderer(object):

    def __init__(self):
        pass

    def plot(self, old_state, action, new_state, reward, done):
        pass

    def render(self, mode=None):
        pass

    def reset(self):
        pass


class OnlineRenderer(Renderer):

    def __init__(self, renderer_provider):
        super().__init__()
        self.q = multiprocessing.Queue()
        self.reset_event = multiprocessing.Event()
        self.worker = multiprocessing.Process(target=render_frame, args=(self.q, self.reset_event, renderer_provider))
        self.startup = True

    def plot(self, old_state, action, new_state, reward, done):
        self.q.put_nowait((old_state, action, new_state, reward, done))
        if self.startup:
            time.sleep(1)
            self.startup = False

    def render(self, mode=None):
        if not self.worker.is_alive():
            self.worker.start()

    def reset(self):
        self.reset_event.set()



