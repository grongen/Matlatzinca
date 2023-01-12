from PyQt5 import QtCore, QtGui
import sys
import traceback
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import time


class Worker(QtCore.QObject, QtCore.QRunnable):
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(tuple)
    result = QtCore.pyqtSignal(object)
    progress = QtCore.pyqtSignal(str)

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        QtCore.QRunnable.__init__(self)

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

        # Add the callback to our kwargs
        self.kwargs["progress_callback"] = self.progress.emit

    @QtCore.pyqtSlot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.error.emit((exctype, value, traceback.format_exc()))
        else:
            if not result is None:
                self.result.emit(result)
                self.finished.emit()
        # finally:
        # self.finished.emit()


class ThreadResizeCanvas(FigureCanvasQTAgg):
    def __init__(self, *args, **kwargs):
        self.lastEvent = False  # store the last resize event's size here

        super().__init__(*args, **kwargs)

        self.resize_thread = QtCore.QThread()
        # Step 3: Create a worker object
        self.sleep_worker = Worker(self.sleep_a_bit)
        self.resize_worker = Worker(self.resize_now)
        # Step 4: Move worker to the thread
        self.sleep_worker.moveToThread(self.resize_thread)
        self.resize_worker.moveToThread(self.resize_thread)
        # Step 5: Connect signals and slots
        self.resize_thread.started.connect(self.sleep_worker.run)
        self.sleep_worker.finished.connect(self.resize_worker.run)
        self.resize_worker.finished.connect(self.resize_thread.quit)

        # self.worker.finished.connect(self.mainwindow.lists_about_to_change.emit)
        # self.resize_thread.finished.connect(self.resize_now)

    def resizeEvent(self, event):

        if not self.lastEvent:
            # at the start of the app, allow resizing to happen.
            super().resizeEvent(event)

        elif not self.resize_thread.isRunning():
            self.resize_thread.start()

        # store the event size for later use
        self.lastEvent = (event.size().width(), event.size().height())

    def sleep_a_bit(self, progress_callback=None):
        time.sleep(0.1)

    def resize_now(self, progress_callback=None):
        newsize = QtCore.QSize(self.lastEvent[0], self.lastEvent[1])
        event = QtGui.QResizeEvent(newsize, QtCore.QSize(1, 1))
        super().resizeEvent(event)
