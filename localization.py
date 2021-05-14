import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets
import sys
from pyqtgraph.Qt import QtGui, QtCore
from random import randint

class Plot2D(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(Plot2D, self).__init__(*args, **kwargs)

        self.plot = pg.PlotWidget()
        self.setCentralWidget(self.plot)

        """hour = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        temperature = [30, 32, 34, 32, 33, 31, 29, 32, 35, 45]"""
        self.x = np.arange(0,5.01, 0.01)
        self.y = [randint(-10,10) for _ in range(len(self.x))]

        # plot data: x, y values
        self.d = self.plot.plot(self.x, self.y)

        ##init the timer for real time plotting
        self.timer = QtCore.QTimer()
        self.timer.setInterval(10) #50ms
        self.timer.timeout.connect(self.update_real_time)
        self.timer.start()

    def update_real_time(self):
        self.x=np.concatenate((self.x[1:], [self.x[-1]+0.01]), axis=None)
        self.y=np.concatenate((self.y[1:], [randint(-10,10)]), axis=None)

        self.d.setData(self.x, self.y)

    def setTitle(self, str):
        self.plot.setTitle(str)

    def setXAxis(self, str, style):
        #styles = {'font-size': '20px'}
        self.plot.setLabel('bottom', str)

    def setYAxis(self, str, style={}, loc='left'):
        self.plot.setLabel(loc, str)

    def grid(self, x=True, y=True):
        self.plot.showGrid(x,y)

    def yLimits(self, lims, e=0):
        self.plot.setYRange(lims[0], lims[1], padding=e)

    def limits(self, lims):
        try:
            if(isinstance(lims, list) == False):
                raise("TypeError: input should be list")
            else:
                lims = [lims[0], lims[1], 0, lims[2]] if(len(lims) == 3) else lims
                lims = [lims[0], lims[1], lims[0], lims[1]] if len(lims) == 2 else lims
                lims = [lims[0], lims[0], lims[0], lims[0]] if len(lims) == 1 else lims
                self.plot.setXRange(lims[0], lims[1], padding=0)
                self.plot.setYRange(lims[2], lims[3], padding=0)
        except Exception as e:
            print(e)

    def clearPlot(self):
        self.plot.clear()


    def start(self):
        if((sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION")):
            QtGui.QGuiApplication.instance().exec_()

def main():
    app = QtWidgets.QApplication(sys.argv)
    main = Plot2D()
    main.setTitle("Dummy figure")
    main.setXAxis("time [h]", {'color':'r', 'font-size':'20px'})
    main.setYAxis("Temperature ")
    main.yLimits([-10, 10], 2)
    #main.limits([0, 3.05, -10, 10])
    main.grid()
    main.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()