from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matlatzinca.ui import widgets
from pathlib import Path


class CondProbSignals(QtCore.QObject):

    # exc_prob_changed = QtCore.pyqtSignal(str, int)
    # exc_prob1_changed = QtCore.pyqtSignal(str)
    # exc_prob2_changed = QtCore.pyqtSignal(str)
    # exc_prob3_changed = QtCore.pyqtSignal(str)

    def __init__(self, dialog):
        super().__init__()
        self.dialog = dialog

    def connect_signals(self):
        self.dialog.lineedits[0].LineEdit.textEdited.connect(lambda s: self.dialog._exc_prob_changed(0, s))
        self.dialog.lineedits[1].LineEdit.textEdited.connect(lambda s: self.dialog._exc_prob_changed(1, s))
        self.dialog.lineedits[2].LineEdit.textEdited.connect(lambda s: self.dialog._exc_prob_changed(2, s))

        # self.dialog.lineedits[1].LineEdit.textEdited.connect(self.signals.exc_prob2_changed.emit)
        # self.dialog.lineedits[2].LineEdit.textEdited.connect(self.signals.exc_prob3_changed.emit)


class ConditionalProbabilitiesDialog(QtWidgets.QDialog):
    """
    Dialog that shows the sensiticity of excluding experts or items
    from the project on the information and calibration score. The
    spread is shown by a box plot for each number of excluded experts
    or items.
    """

    def __init__(self, parent):
        """
        Constructor
        """
        super().__init__()

        self.data_path = parent.mainwindow.data_path
        self.load_data()
        self.rho = 0.0

        self.icon = parent.mainwindow.icon
        self.signals = CondProbSignals(self)
        self.construct_dialog()
        self.signals.connect_signals()

        for i, p in enumerate([0.05, 0.5, 0.95]):
            self.lineedits[i].set_value(p)
            self.update_curve(i, p)

        self.update_line(self.rho)

        # Increase or decrease fontsize to match changes in mainwindow
        for w in self.children():
            if isinstance(w, QtWidgets.QWidget):
                font = w.font()
                font.setPointSize(max(1, font.pointSize() + parent.mainwindow.font_increment))
                w.setFont(font)

    def load_data(self):
        with (self.data_path / "cond_prob_rho.npy").open("rb") as f:
            self.arr = np.load(f)

    def update_curve(self, i: int, ovkans: float) -> None:
        ranks = np.linspace(-0.9999, 0.9999, 201)
        rhos = 2 * np.sin((np.pi / 6) * ranks)
        ovkansen = np.linspace(0.01, 0.99, 99)
        ip = np.argmin(np.absolute(ovkansen - ovkans))
        self.lines[i].set_data(ranks, self.arr[:, ip])
        self.axs[i].set_ylabel(self._get_exc_string(ovkans))
        self.canvasses[i].draw_idle()

    @staticmethod
    def _get_exc_string(p):
        return f"$P(F(X_1) > {p:.2g}  |  F(X_2) > {p:.2g})$"

    def construct_dialog(self):
        """
        Constructs the widget.
        """

        self.setWindowTitle("Conditional probabilities")
        self.setWindowIcon(self.icon)
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)

        # Create figure
        self.figures = []
        self.axs = []
        self.canvasses = []
        self.lines = []
        self.vlines = []
        self.hlines = []
        self.lineedits = []
        self.texts = []

        validator = QtGui.QDoubleValidator()
        validator.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
        validator.setRange(0.01, 0.99, 2)
        self.parameterinput = []

        bgcolor = self.palette().color(self.backgroundRole()).name()
        layout = []

        for i in range(3):
            fig, ax = plt.subplots()
            # Set background color
            fig.patch.set_facecolor(bgcolor)
            # Add canvas
            canvas = FigureCanvasQTAgg(fig)
            # canvas.mpl_connect("draw_event", self._draw_event)
            self.canvasses.append(canvas)
            self.figures.append(fig)
            self.axs.append(ax)
            self.lines.append(ax.plot([], [], color="C0", lw=1.5)[0])
            self.vlines.append(ax.axvline(self.rho, ls=":", color="k", lw=1))
            self.hlines.append(ax.axhline(0.5, ls="--", color="C3", lw=1))
            self.texts.append(ax.text(x=1.0, y=0.505, s=f"$P = 0.0$", ha="right", va="bottom", color="C3"))

            underscore1 = "\u2081"
            pi = widgets.ParameterInputLine(label=f"F(X{underscore1}) = ", validator=validator)
            font = pi.Label.font()
            font.setItalic(True)
            pi.Label.setFont(font)

            self.lineedits.append(pi)
            layout.append(widgets.VLayout([canvas, pi]))

        for ax in self.axs:
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.tick_params(axis="y", color="0.75")
            ax.tick_params(axis="x", color="0.75")
            ax.axis((-1, 1, 0, 1))
            ax.grid()
            ax.set_ylabel(self._get_exc_string(0.0))

        for fig in self.figures:
            fig.set_tight_layout("tight")

        # Create slider
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(-99)
        self.slider.setMaximum(99)
        self.slider.setValue(int(self.rho * 100))
        self.slider.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.slider.setTickInterval(1)

        self.corr_label = QtWidgets.QLabel(f"{self.rho:.2f}")
        self.corr_label.setMinimumWidth(50)
        self.slider.valueChanged.connect(lambda x: self.corr_label.setText(f"{x/100:.2f}"))
        self.slider.valueChanged.connect(lambda x: self.update_line(x))
        # self.slider.sliderReleased.connect(lambda: self.update_line(self.slider.value()))

        self.setLayout(
            widgets.VLayout(
                [
                    widgets.HLayout(layout),
                    widgets.HLayout(
                        [
                            QtWidgets.QLabel("Rank correlation:"),
                            self.corr_label,
                            QtWidgets.QLabel("-1"),
                            self.slider,
                            QtWidgets.QLabel("+1"),
                        ]
                    ),
                    widgets.HLine(),
                    widgets.HLayout(["stretch", QtWidgets.QPushButton("Close", clicked=self.close)]),
                ]
            )
        )

        self.resize(900, 400)

    # def _draw_event(self, evt):
    # """after drawing, grab the new background"""
    # for ax, canvas in zip(self.axs, self.canvasses):
    # self.bg = canvas.copy_from_bbox(ax.bbox)

    def _exc_prob_changed(self, i: int, value: str):
        if not value.replace(".", "").isdecimal():
            return None
        else:
            ovkans = float(value)
            self.update_curve(i, ovkans)

    def update_line(self, value: int):
        for line, vline, hline, text in zip(self.lines, self.vlines, self.hlines, self.texts):
            r = value / 100
            vline.set_xdata([r, r])
            p = np.interp(r, *line.get_data())
            hline.set_ydata([p, p])
            text.set_text(f"$P = {p:.3f}$")
            text.set_y(p)

        for canvas in self.canvasses:
            canvas.draw_idle()
