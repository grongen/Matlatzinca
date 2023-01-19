import logging
import subprocess
import sys
import traceback
from pathlib import Path

from matlatzinca import __version__
from matlatzinca.core.models import Node
from matlatzinca.core.project import Project
from matlatzinca.ui import widgets
from matlatzinca.ui.dialogs import NotificationDialog, get_icon
from matlatzinca.ui.graph import GraphWidget
from matlatzinca.ui.logging import initialize_logger
from matlatzinca.ui.matrix import MatrixWidget
from matlatzinca.ui.nodeedge import NodeEdgeWidget
from PyQt5 import Qt, QtCore, QtGui, QtWidgets

logger = logging.getLogger(__name__)

# TODO: Square icon


class Signals(QtCore.QObject):

    lists_about_to_change = QtCore.pyqtSignal()
    lists_changed = QtCore.pyqtSignal()

    node_about_to_be_added = QtCore.pyqtSignal()
    node_coordinate_selected = QtCore.pyqtSignal(tuple)
    node_added = QtCore.pyqtSignal(tuple, Node)

    node_about_to_be_removed = QtCore.pyqtSignal(str)
    node_removed = QtCore.pyqtSignal(str)

    # sampling_order_about_to_change = QtCore.pyqtSignal(int, int)
    # sampling_order_changed = QtCore.pyqtSignal()

    node_order_about_to_change = QtCore.pyqtSignal(int, int)
    node_order_changed = QtCore.pyqtSignal()

    parent_order_about_to_change = QtCore.pyqtSignal(int, int)
    parent_order_changed = QtCore.pyqtSignal()

    name_about_to_change = QtCore.pyqtSignal(str, str)
    name_changed = QtCore.pyqtSignal()

    edge_about_to_be_added = QtCore.pyqtSignal()
    edge_nodes_selected = QtCore.pyqtSignal(object)
    edge_added = QtCore.pyqtSignal(str, str)

    edge_about_to_be_removed = QtCore.pyqtSignal(str, str)
    edge_removed = QtCore.pyqtSignal(str, str)

    edge_about_to_be_reversed = QtCore.pyqtSignal(str, str)
    edge_reversed = QtCore.pyqtSignal(str, str)

    observed_correlation_about_to_change = QtCore.pyqtSignal(str, str, str)
    correlation_about_to_change = QtCore.pyqtSignal(str, str, str)
    correlation_changed = QtCore.pyqtSignal()

    on_click_delete_about_to_happen = QtCore.pyqtSignal()
    stop_running_click_threads = QtCore.pyqtSignal()

    set_graph_message = QtCore.pyqtSignal(str)

    font_changed = QtCore.pyqtSignal()

    set_window_modified = QtCore.pyqtSignal(bool)

    selected = QtCore.pyqtSignal(object)
    nodeview_row_changed = QtCore.pyqtSignal(str, str)
    edgeview_row_changed = QtCore.pyqtSignal(tuple, tuple)

    def __init__(self, parent):
        super().__init__()
        self.mainwindow = parent

    def connect(self) -> None:

        # Change edge labels
        self.edge_added.connect(self.mainwindow.graph_widget.change_edge_labels)
        self.node_removed.connect(self.mainwindow.graph_widget.change_edge_labels)
        self.edge_removed.connect(self.mainwindow.graph_widget.change_edge_labels)
        self.edge_reversed.connect(self.mainwindow.graph_widget.change_edge_labels)
        # self.sampling_order_changed.connect(self.mainwindow.graph_widget.change_edge_labels)
        self.node_order_changed.connect(self.mainwindow.graph_widget.change_edge_labels)
        self.parent_order_changed.connect(self.mainwindow.graph_widget.change_edge_labels)

        # Connect window modified signals here
        # Connect node related
        self.node_added.connect(lambda: self.set_window_modified.emit(True))
        self.node_removed.connect(lambda: self.set_window_modified.emit(True))
        self.name_changed.connect(lambda: self.set_window_modified.emit(True))
        # self.sampling_order_changed.connect(lambda: self.set_window_modified.emit(True))
        self.node_order_changed.connect(lambda: self.set_window_modified.emit(True))
        # Connect edge related
        self.edge_added.connect(lambda: self.set_window_modified.emit(True))
        self.edge_removed.connect(lambda: self.set_window_modified.emit(True))
        self.edge_reversed.connect(lambda: self.set_window_modified.emit(True))
        self.correlation_changed.connect(lambda: self.set_window_modified.emit(True))
        self.parent_order_changed.connect(lambda: self.set_window_modified.emit(True))

        # Connect update correlation matrix signals
        # self.node_added.connect(lambda: self.mainwindow.matrix_widget.update_correlation_matrix)
        # self.node_removed.connect(lambda: self.mainwindow.matrix_widget.update_correlation_matrix)
        # self.node_order_changed.connect(lambda: self.mainwindow.matrix_widget.update_correlation_matrix)
        # self.name_changed.connect(lambda: self.mainwindow.matrix_widget.update_correlation_matrix)
        # self.edge_removed.connect(lambda: self.mainwindow.matrix_widget.update_correlation_matrix)
        # self.edge_reversed.connect(lambda: self.mainwindow.matrix_widget.update_correlation_matrix)
        # self.correlation_changed.connect(lambda: self.mainwindow.matrix_widget.update_correlation_matrix)

        # Connect print signals
        self.selected.connect(lambda s: logger.info(f"Clicked {'edge' if isinstance(s, tuple) == 1 else 'node'} {s}."))


class MainWindow(QtWidgets.QMainWindow):
    """
    Main UI widget of Anduryl.
    """

    def __init__(self, app):
        """
        Constructer. Adds project and sets general settings.
        """
        super().__init__()

        self.app = app
        # custom_font = QtGui.QFont()
        # custom_font.setPointSize(10)
        # app.setFont(custom_font)  # , "QPushButton")
        # app.setFont(custom_font, "QTableView")
        # app.setFont(custom_font, "QTableView")

        self.appsettings = QtCore.QSettings("Matlatzinca")

        # font = self.app.font()
        # font.setPixelSize(18)

        self.setAcceptDrops(True)

        self.setWindowTitle("Matlatzinca")

        initialize_logger(console=True)

        self.profiling = False

        self.data_path = self.get_data_path()

        self.icon = QtGui.QIcon(str(self.data_path / "icon.ico"))
        self.setWindowIcon(self.icon)

        # Add signals
        self.signals = Signals(self)
        self.signals.set_window_modified.connect(self.setWindowModified)

        # Add keyboard shortcuts
        self.shortcut_add_node = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+1"), self)
        self.shortcut_add_node.activated.connect(self.signals.node_about_to_be_added)

        self.shortcut_add_edge = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+2"), self)
        self.shortcut_add_edge.activated.connect(self.signals.edge_about_to_be_added)

        # Add project
        self.project = Project(self)
        self.update_projectname()

        self.setCursor(Qt.Qt.ArrowCursor)

        self.bordercolor = "lightgrey"
        # Construct user interface
        self.init_ui()
        self.signals.connect()

        # Keep track of font size increment, for opening new windows
        self.font_increment = 0

        def test_exception_hook(exctype, value, tback):
            """
            Function that catches errors and gives a Notification
            instead of a crashing application.
            """
            sys.__excepthook__(exctype, value, tback)
            self.setCursorNormal()
            NotificationDialog(
                text="\n".join(traceback.format_exception_only(exctype, value)),
                severity="critical",
                details="\n".join(traceback.format_tb(tback)),
            )

        sys.excepthook = test_exception_hook

    # def resizeEvent(self, event):
    #     print("Window has been resized")
    #     QtWidgets.QMainWindow.resizeEvent(self, event)

    def get_data_path(self) -> Path:
        # In case of PyInstaller exe
        if getattr(sys, "frozen", False):
            application_path = sys._MEIPASS
            data_path = Path(application_path) / "data"

        # In case of regular python
        else:
            application_path = Path(__file__).resolve().parent
            data_path = application_path / ".." / "data"

        return data_path

    def setCursorNormal(self):
        """
        Changes cursor (back) to normal cursor.
        """
        Qt.QApplication.restoreOverrideCursor()

    def setCursorWait(self):
        """
        Changes cursor to waiting cursor.
        """
        Qt.QApplication.setOverrideCursor(Qt.QCursor(QtCore.Qt.WaitCursor))
        Qt.QApplication.processEvents()

    def init_ui(self):
        """
        Construct UI, by splitting the main window and adding the
        different widgets with tables.
        """
        mainsplitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        self.setCentralWidget(mainsplitter)

        # GraphWidget
        self.graph_widget = GraphWidget(self)

        # MatrixWidget
        self.matrix_widget = MatrixWidget(self)

        # Lists of Nodes and Edges
        self.nodes_edges_widget = NodeEdgeWidget(self)

        self.rightsplitter = widgets.LogoSplitter(
            self.matrix_widget, self.nodes_edges_widget, "Correlation Matrix", "Nodes and Edges"
        )
        self.rightsplitter.setSizes([200, 200])

        mainsplitter.addWidget(self.graph_widget)
        mainsplitter.addWidget(self.rightsplitter)
        mainsplitter.setSizes([500, 500])

        self.init_menubar()

        mainsplitter.setStyleSheet("QSplitter::handle{background:white}")
        self.rightsplitter.setStyleSheet("QSplitter::handle{background:white}")

        self.setGeometry(300, 200, 1200, 650)

        self.show()

    def change_font_size(self, increment):

        self.nodes_edges_widget.nodemodel.layoutAboutToBeChanged.emit()
        self.nodes_edges_widget.edgemodel.layoutAboutToBeChanged.emit()

        sizes = []
        for w in self.app.allWidgets():
            font = w.font()
            size = font.pointSize() + increment
            font.setPointSize(max(1, size))
            sizes.append(size)
            w.setFont(font)

        # Keep track of font increment for opening new windows in the right scale
        # Only if the smallest item has a >= 1 size, the increment is added, to prevent the value getting out of bounds
        if min(sizes) >= 1:
            self.font_increment += increment

        self.signals.font_changed.emit()

        self.nodes_edges_widget.nodemodel.layoutChanged.emit()
        self.nodes_edges_widget.edgemodel.layoutChanged.emit()

    def update_projectname(self, name=None):
        """
        Updates window title after a project has been loaded

        Parameters
        ----------
        name : str, optional
            Project name, by default None
        """
        if name is None:
            self.setWindowTitle("Matlatzinca [*]")
            self.appsettings.setValue("currentproject", "")
        else:
            self.setWindowTitle(f"Matlatzinca - {name} [*]")
            self.appsettings.setValue("currentproject", name)

    def init_menubar(self):
        """
        Constructs the menu bar.
        """

        menubar = self.menuBar()

        new_action = QtWidgets.QAction(self.style().standardIcon(QtWidgets.QStyle.SP_FileIcon), "New", self)
        new_action.setShortcut(QtGui.QKeySequence.New)
        new_action.setStatusTip("Create a new project")
        new_action.triggered.connect(self.project.new)

        openAction = QtWidgets.QAction(self.style().standardIcon(QtWidgets.QStyle.SP_DirOpenIcon), "Open", self)
        openAction.setStatusTip("Open project")
        openAction.setShortcut(QtGui.QKeySequence.Open)
        openAction.triggered.connect(self.project.open)

        saveAction = QtWidgets.QAction(self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton), "Save", self)
        saveAction.setStatusTip("Save project")
        saveAction.setShortcut(QtGui.QKeySequence.Save)
        saveAction.triggered.connect(self.project.save)

        saveAsAction = QtWidgets.QAction(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton), "Save as", self
        )
        saveAsAction.setStatusTip("Save project as...")
        saveAsAction.setShortcut("Ctrl+Shift+S")
        saveAsAction.triggered.connect(self.project.save_as)

        exitAction = QtWidgets.QAction(
            self.style().standardIcon(QtWidgets.QStyle.SP_TitleBarCloseButton), "Exit", self
        )
        exitAction.setShortcut("Ctrl+Q")
        exitAction.setStatusTip("Close Matlatzinca")
        exitAction.triggered.connect(self.close)

        file_menu = menubar.addMenu("&File")
        file_menu.addAction(new_action)
        file_menu.addAction(openAction)
        file_menu.addSeparator()
        file_menu.addAction(saveAction)
        file_menu.addAction(saveAsAction)
        file_menu.addSeparator()
        file_menu.addAction(exitAction)

        view_menu = menubar.addMenu("&View")
        view_menu.addAction("Increase UI font", lambda: self.change_font_size(1), QtGui.QKeySequence("Ctrl+="))
        view_menu.addAction("Decrease UI font", lambda: self.change_font_size(-1), QtGui.QKeySequence("Ctrl+-"))
        view_menu.addSeparator()

        def _increase_dpi():
            self.graph_widget.increase_dpi(factor=1.2)
            self.matrix_widget.increase_dpi(factor=1.2)

        def _decrease_dpi():
            self.graph_widget.decrease_dpi(factor=1.2)
            self.matrix_widget.decrease_dpi(factor=1.2)

        view_menu.addAction("Increase graph scale", _increase_dpi, QtGui.QKeySequence("Ctrl+Shift+="))
        view_menu.addAction("Decrease graph scale", _decrease_dpi, QtGui.QKeySequence("Ctrl+Shift+-"))

        export_menu = menubar.addMenu("&Export")
        export_R_menu = export_menu.addMenu("&Correlation Matrix")
        export_R_menu.addAction("To CSV", lambda: self.project.export("correlation_matrix", "csv"))
        export_R_menu.addAction("To clipboard", lambda: self.project.export("correlation_matrix", "clipboard"))

        export_nodes_menu = export_menu.addMenu("&Nodes")
        export_nodes_menu.addAction("To CSV", lambda: self.project.export("nodes", "csv"))
        export_nodes_menu.addAction("To clipboard", lambda: self.project.export("nodes", "clipboard"))

        export_nodes_menu = export_menu.addMenu("&Edges")
        export_nodes_menu.addAction("To CSV", lambda: self.project.export("edges", "csv"))
        export_nodes_menu.addAction("To clipboard", lambda: self.project.export("edges", "clipboard"))

        help_menu = menubar.addMenu("&Help")
        doc_action = QtWidgets.QAction(
            self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView), "Documentation", self
        )
        doc_action.setStatusTip("Open Anduryl documentation")
        doc_action.triggered.connect(self.open_documentation)
        help_menu.addAction(doc_action)

        about_action = QtWidgets.QAction(QtGui.QIcon(), "Version", self)
        about_action.triggered.connect(self.open_about)
        help_menu.addAction(about_action)

    def open_about(self):
        text = f"Version: {__version__}"
        Qt.QMessageBox.about(self, "Matlatzinca version", text)

    def open_documentation(self):

        # In case of PyInstaller exe
        if getattr(sys, "frozen", False):
            application_path = sys._MEIPASS
            indexpath = application_path / "doc" / "index.html"

        # In case of regular python
        else:
            application_path = Path(__file__).resolve().parent
            indexpath = application_path / '..' / '..' / "doc" / 'build' / 'html' / "index.html"

        # Open index html
        subprocess.Popen(str(indexpath), shell=True)
