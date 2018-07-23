from visdom import Visdom
import numpy

class LinePlot(object):
    def __init__(self, title, port=8082):
        self.viz = Visdom(port=port)
        self.windows = {}
        self.title = title

    def register_plot(self, name, xlabel, ylabel):
        win = self.viz.line(
            X=numpy.zeros([1]),
            Y=numpy.zeros([1]),
            opts=dict(title=self.title, markersize=5, xlabel=xlabel, ylabel=ylabel)
        )
        self.windows[name] = win

    def update_plot(self, name, x, y):
        self.viz.line(
            X=numpy.array([x]),
            Y=numpy.array([y]),
            win=self.windows[name],
            update='append'
        )
