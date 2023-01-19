__author__ = """Guus Rongen"""
__email__ = "g.w.f.rongen@tudelft.nl"
__version__ = "1.0.0"

if __name__ == "__main__":

    # Import PyQt modules
    from PyQt5 import QtWidgets, QtCore
    import sys

    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)

    app.setApplicationVersion(__version__)

    # Import GUI
    # from matlatzinca.core.main import Project
    from matlatzinca.ui.main import MainWindow

    # Create main window
    ex = MainWindow(app)

    # Open project
    if len(sys.argv) > 1:
        ex.project.open(fname=sys.argv[1])
        ex.setCursorNormal()

    sys.exit(app.exec_())
