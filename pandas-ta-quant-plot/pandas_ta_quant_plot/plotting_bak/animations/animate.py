import io
from contextlib import redirect_stdout
from datetime import datetime
from typing import Callable


def plot_animation(index, make_frame: Callable[[datetime], 'Figure'], fps=2, **kwargs):
    from moviepy.video.VideoClip import DataVideoClip
    from moviepy.video.io.html_tools import ipython_display

    def repr_html(clip):
        f = io.StringIO()
        with redirect_stdout(f):
            return ipython_display(clip)._repr_html_()

    animation = DataVideoClip(index, make_frame, fps=fps, **kwargs)

    # bind the jupyter extension to the animation and return
    setattr(DataVideoClip, 'display', ipython_display)
    setattr(DataVideoClip, '_repr_html_', repr_html)
    return animation
