import matplotlib.pyplot as plt
import numpy as np
from matlatzinca.ui import widgets, conditional
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.collections import EllipseCollection
from PyQt5 import QtWidgets

plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams["axes.labelsize"] = 9
plt.rcParams["axes.titlesize"] = 9
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["legend.handletextpad"] = 0.4
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["legend.labelspacing"] = 0.2
plt.rcParams["legend.fancybox"] = False

plt.rcParams["font.size"] = 9

plt.rcParams["lines.linewidth"] = 1.5
plt.rcParams["lines.markeredgewidth"] = 0
plt.rcParams["lines.markersize"] = 5

plt.rcParams["figure.dpi"] = 100


class MatrixWidget(QtWidgets.QWidget):
    """ """

    def __init__(self, mainwindow):
        """
        Constructor
        """
        super().__init__()

        self.construct_widget()

        self.mainwindow = mainwindow
        self.signals = mainwindow.signals
        self.project = mainwindow.project

        self.signals.node_added.connect(self.update_correlation_matrix)
        self.signals.node_removed.connect(self.update_correlation_matrix)
        self.signals.edge_removed.connect(self.update_correlation_matrix)
        self.signals.edge_reversed.connect(self.update_correlation_matrix)
        self.signals.correlation_changed.connect(self.update_correlation_matrix)
        self.signals.node_order_changed.connect(self.update_correlation_matrix)
        self.signals.name_changed.connect(self.update_correlation_matrix)

        # self.signals.nodeview_row_changed.connect(self.select_node)
        # self.signals.edgeview_row_changed.connect(self.select_edge)

    def update_correlation_matrix(self):

        self.corrmatrix.update_data(R=self.project.bn.R, labels=self.project.bn._node_names)
        self.corrmatrix.update_plot()

        # self.ax.imshow(self.project.bn.R, vmin=-1.0, vmax=1.0, cmap="RdBu")
        # self.update_tick_labels()

    def construct_widget(self):
        """
        Constructs the widget.
        """

        # Create figure
        self.figure = plt.figure(constrained_layout=True)
        self.ax = plt.subplot2grid(shape=(1, 25), loc=(0, 0), colspan=24)
        self.cax = plt.subplot2grid(shape=(1, 25), loc=(0, 24), colspan=1)
        self.corrmatrix = SquareCircleCorrelationMatrix(self.ax, self.cax)

        # Set background color
        bgcolor = self.palette().color(self.backgroundRole()).name()
        self.figure.patch.set_facecolor(bgcolor)
        self.ax.set_facecolor(bgcolor)
        # self.figure.tight_layout()

        self.ax.spines["right"].set_visible(False)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["left"].set_visible(False)
        self.ax.spines["bottom"].set_visible(False)

        self.ax.tick_params(axis="y", length=0.0)
        self.ax.tick_params(axis="x", length=0.0, rotation=90)

        self.ax.set(aspect=1.0, xticks=[], yticks=[])
        # self.ax.axis((0, 1, 0, 1))

        # Add canvas
        self.canvas = FigureCanvasQTAgg(self.figure)
        # print(self.canvas._dpi_ratio)

        # Buttons
        cmaps = [
            "PiYG",
            "PRGn",
            "BrBG",
            "PuOr",
            "RdGy",
            "RdBu",
            "RdYlBu",
            "RdYlGn",
            "Spectral",
            "coolwarm",
            "bwr",
            "seismic",
            "viridis",
            "plasma",
            "cividis",
            "Greys",
            "Purples",
            "Blues",
            "Greens",
            "Oranges",
            "Reds",
            "YlOrBr",
            "YlOrRd",
            "OrRd",
            "PuRd",
            "RdPu",
            "BuPu",
            "GnBu",
            "PuBu",
            "YlGnBu",
            "PuBuGn",
            "BuGn",
            "YlGn",
        ]
        width = widgets.get_width("Show coefs.:")
        self.cmap_combobox = widgets.ComboboxInputLine(
            "Colormap:", labelwidth=width, items=cmaps, default="RdBu"
        )
        self.cmap_combobox.combobox.setFixedWidth(80)
        self.cmap_combobox.combobox.currentIndexChanged.connect(self.set_colormap)

        self.cmap_invert_checkbox = widgets.CheckBoxInput("Invert:", labelwidth=width)
        self.cmap_invert_checkbox.checkbox.stateChanged.connect(self.set_colormap)

        self.cmin_parameter = widgets.ExtendedLineEdit("Min. value:", labelwidth=width)
        self.cmin_parameter.set_value(-1.0)
        self.cmin_parameter.LineEdit.setFixedWidth(80)
        self.cmin_parameter.LineEdit.editingFinished.connect(self.set_cmin)

        self.cmax_parameter = widgets.ExtendedLineEdit("Max. value:", labelwidth=width)
        self.cmax_parameter.set_value(1.0)
        self.cmax_parameter.LineEdit.setFixedWidth(80)
        self.cmin_parameter.LineEdit.editingFinished.connect(self.set_cmax)

        self.show_labels_checkbox = widgets.CheckBoxInput("Show coefs.:", labelwidth=width)
        self.show_labels_checkbox.checkbox.stateChanged.connect(self.disp_correlations)
        self.show_labels_checkbox.set_value(True)

        self.plot_conditional_button = QtWidgets.QPushButton(
            "Plot conditional\nprobabilities",
            clicked=self.plot_conditional_probabilities,
        )

        self.buttonlayout = widgets.VLayout(
            [
                self.cmap_combobox,
                self.cmap_invert_checkbox,
                self.cmin_parameter,
                self.cmax_parameter,
                widgets.HLine(),
                self.show_labels_checkbox,
                widgets.HLine(),
                self.plot_conditional_button,
                "stretch",
            ]
        )
        # self.buttonlayout.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)

        self.setLayout(widgets.HLayout([self.canvas, self.buttonlayout], stretch=[100, 1]))

    def plot_conditional_probabilities(self):
        self.plot_conditional_dialog = conditional.ConditionalProbabilitiesDialog(self)
        self.plot_conditional_dialog.exec_()

    def disp_correlations(self):
        display = self.show_labels_checkbox.get_value()
        self.corrmatrix.set_texts_visible(display)

    # def select_node(self, oldnode, newnode):
    # pass

    # def select_edge(self, newedge, oldedge):
    # self.corrmatrix.select_edge(*newedge)

    def set_colormap(self):
        cmap = self.cmap_combobox.get_value()
        add = "_r" if self.cmap_invert_checkbox.get_value() else ""
        self.corrmatrix.cmap = cmap + add
        self.corrmatrix.cc.set_cmap(cmap + add)
        self.corrmatrix.cb.update_normal(self.corrmatrix.cc)
        self.canvas.draw_idle()

    def set_cmin(self):
        value = float(self.cmin_parameter.get_value())
        current = self.corrmatrix.cc.get_clim()
        self.corrmatrix.cc.set_clim(value, current[1])
        self.corrmatrix.cb.update_normal(self.corrmatrix.cc)
        self.canvas.draw_idle()

    def set_cmax(self):
        value = float(self.cmax_parameter.get_value())
        current = self.corrmatrix.cc.get_clim()
        self.corrmatrix.cc.set_clim(current[0], value)
        self.corrmatrix.cb.update_normal(self.corrmatrix.cc)
        self.canvas.draw_idle()

    def increase_dpi(self, factor=1.1):
        current_dpi = self.figure.get_dpi()
        self._set_figure_dpi(current_dpi * factor)

    def decrease_dpi(self, factor=1.1):
        current_dpi = self.figure.get_dpi()
        self._set_figure_dpi(current_dpi / factor)

    def _set_figure_dpi(self, new_dpi):
        current_size = self.figure.get_size_inches()
        current_dpi = self.figure.get_dpi()

        self.figure.set_dpi(new_dpi)
        self.figure.set_size_inches(*(current_size * current_dpi / new_dpi))
        self.canvas.draw_idle()


class TriangularCircleCorrelationMatrix:
    def __init__(self, ax, cax):

        self.ax = ax
        self.cax = cax
        (self.grid,) = self.ax.plot([], [], color="grey", lw=0.5)
        self.cc = None
        self.cb = None
        self.texts = []

        self.R = np.ndarray(shape=(0, 0), dtype=np.float64)
        self.n = 0
        self.labels = []
        self.show_correlations = True
        self.cmap = "RdBu"

    def update_data(self, R, labels):
        self.R = R
        self.n = len(R)
        self.labels = labels

    def update_plot(self):

        # Calculate positions
        rs = self.R[np.triu_indices(self.n, 1)]
        xy = np.stack(np.triu_indices(self.n, 1), axis=1)
        sizes = np.absolute(rs) ** 0.5

        # Remove old collection if present
        if self.cc is not None:
            self.cc.remove()

        # Create new collection
        self.cc = EllipseCollection(
            sizes,
            sizes,
            np.zeros_like(sizes),
            offsets=xy,
            units="x",
            transOffset=self.ax.transData,
            cmap=self.cmap,
            clim=(-1, 1),
        )
        self.cc.set_array(rs)
        self.ax.add_collection(self.cc)

        # Add colorbar (only first time, but in update function because collection is needed)
        if self.cb is None:
            self.cb = plt.colorbar(self.cc, cax=self.cax)
            self.cb.set_label("Rank correlation [-]")

        m = 0.05
        self.ax.axis((-0.5 - m, self.n - 1.5 + m, self.n - 0.5 + m, 0.5 - m))

        # Update texts
        for text in self.texts[::-1]:
            text.remove()
            self.texts.remove(text)
        for pos, r in zip(xy, rs):
            self.texts.append(self.ax.text(*pos, f"{r:.2f}", ha="center", va="center"))
        self.set_texts_visible()

        # Update tick labels
        self._update_ticklabels()

        # Update grid (only needed on size change)
        self._plot_tri_grid()

    def set_texts_visible(self, visible: bool = None) -> None:
        if visible is not None:
            self.show_correlations = visible

        for text in self.texts:
            text.set_visible(self.show_correlations)

        self.ax.figure.canvas.draw_idle()

    def _update_ticklabels(self) -> None:

        self.ax.set_xticks(list(range(self.n - 1)))
        self.ax.set_xticklabels(self.labels[:-1])
        self.ax.set_yticks(list(range(1, self.n)))
        self.ax.set_yticklabels(self.labels[1:])

    def _plot_tri_grid(self) -> None:
        """Plot grid around lower left triangle of matrix"""

        triux, triuy = np.stack(np.triu_indices(self.n + 1)).astype(float)
        pos = np.where(np.diff(triux) != 0)[0] + 1
        triux = np.insert(triux, pos, np.nan)[1:]
        triuy = np.insert(triuy, pos, np.nan)[1:]

        trilx, trily = np.stack(np.tril_indices(self.n + 1)).astype(float)
        pos = np.where(np.diff(trilx) != 0)[0] + 1
        trilx = np.insert(trilx, pos, np.nan)[:-1]
        trily = np.insert(trily, pos, np.nan)[:-1]

        trix = np.concatenate([triux, [np.nan], trily]) - 0.5
        triy = np.concatenate([triuy, [np.nan], trilx]) - 0.5

        self.grid.set_data(trix, triy)


class SquareCircleCorrelationMatrix:
    def __init__(self, ax, cax):

        self.ax = ax
        self.cax = cax
        (self.grid,) = self.ax.plot([], [], color="grey", lw=0.5)
        self.cc = None
        self.cb = None
        self.texts = []

        # (self.edge_box,) = self.ax.plot([], [], color="red", lw=1.0, animated=True)

        self.R = np.ndarray(shape=(0, 0), dtype=np.float64)
        self.n = 0
        self.labels = []
        self.show_correlations = True
        self.cmap = "RdBu"

        # self.ax.figure.canvas.mpl_connect("resize_event", self.save_background)

    def update_data(self, R, labels):
        self.R = R
        self.n = len(R)
        self.labels = labels

    def update_plot(self):

        # Calculate positions
        indices = np.where(np.ones(self.R.shape, dtype=bool))
        # noneye = np.where(~np.eye(self.R.shape[0], dtype=bool))
        rs = self.R[indices]
        # xx, yy = np.indices(self.R.shape)
        xy = np.stack(indices, axis=1)
        eye = xy[:, 0] == xy[:, 1]

        sizes = np.absolute(rs) ** 0.5

        # Remove old collection if present
        if self.cc is not None:
            self.cc.remove()

        # Create new collection
        self.cc = EllipseCollection(
            sizes[~eye],
            sizes[~eye],
            np.zeros_like(sizes[~eye]),
            offsets=xy[~eye],
            units="x",
            transOffset=self.ax.transData,
            cmap=self.cmap,
            clim=(-1, 1),
        )
        self.cc.set_array(rs[~eye])
        self.ax.add_collection(self.cc)

        # Add colorbar (only first time, but in update function because collection is needed)
        if self.cb is None:
            self.cb = plt.colorbar(self.cc, cax=self.cax)
            self.cb.set_label("Rank correlation [-]")

        m = 0.05
        self.ax.axis((-0.5 - m, self.n - 0.5 + m, -0.5 - m, self.n - 0.5 + m))

        # Update texts
        for text in self.texts[::-1]:
            text.remove()
            self.texts.remove(text)
        for pos, r in zip(xy, rs):
            self.texts.append(self.ax.text(*pos, f"{r:.2f}", ha="center", va="center"))
        self.set_texts_visible()

        # Update tick labels
        self._update_ticklabels()

        # Update grid (only needed on size change)
        self._plot_tri_grid()

        # Save as background
        # fig = self.ax.figure
        # fig.canvas.draw()
        # self.save_background()

    # def save_background(self, event=None) -> None:
    # self.bg = self.ax.figure.canvas.copy_from_bbox(self.ax.figure.bbox)

    def set_texts_visible(self, visible: bool = None) -> None:
        if visible is not None:
            self.show_correlations = visible

        for text in self.texts:
            text.set_visible(self.show_correlations)

        self.ax.figure.canvas.draw_idle()

    def _update_ticklabels(self) -> None:

        self.ax.set_xticks(list(range(self.n)))
        self.ax.set_xticklabels(self.labels)
        self.ax.set_yticks(list(range(self.n)))
        self.ax.set_yticklabels(self.labels)

    def _plot_tri_grid(self) -> None:
        """Plot grid around lower left triangle of matrix"""

        x, y = np.indices((self.R.shape[0] + 1, self.R.shape[1] + 1))
        x = (x.astype(float) - 0.5).reshape(-1)
        y = (y.astype(float) - 0.5).reshape(-1)
        pos = np.where(np.diff(x) != 0)[0] + 1
        x = np.insert(x, pos, np.nan)
        y = np.insert(y, pos, np.nan)

        # trilx, trily = np.stack(np.tril_indices(self.n + 1)).astype(float)
        # pos = np.where(np.diff(trilx) != 0)[0] + 1
        # trilx = np.insert(trilx, pos, np.nan)[:-1]
        # trily = np.insert(trily, pos, np.nan)[:-1]

        xtot = np.concatenate([x, [np.nan], y])
        ytot = np.concatenate([y, [np.nan], x])

        self.grid.set_data(xtot, ytot)

    # def select_edge(self, parent, child):
    #     pos1 = self.labels.index(parent)
    #     pos2 = self.labels.index(child)

    #     fig = self.ax.figure

    #     fig.canvas.restore_region(self.bg)
    #     self.edge_box.set_data(
    #         np.concatenate(
    #             [
    #                 pos1 + np.array([-0.5, 0.5, 0.5, -0.5, -0.5]),
    #                 [np.nan],
    #                 pos2 + np.array([-0.5, 0.5, 0.5, -0.5, -0.5]),
    #             ]
    #         ),
    #         np.concatenate(
    #             [
    #                 pos2 + np.array([-0.5, -0.5, 0.5, 0.5, -0.5]),
    #                 [np.nan],
    #                 pos1 + np.array([-0.5, -0.5, 0.5, 0.5, -0.5]),
    #             ]
    #         ),
    #     )
    #     self.ax.draw_artist(self.edge_box)
    #     fig.canvas.blit(fig.bbox)
    #     # fig.canvas.draw_idle()
