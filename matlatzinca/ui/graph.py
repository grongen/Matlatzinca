import logging
import time
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from matlatzinca.core.models import Node
from matlatzinca.core.threads import Worker
from matlatzinca.ui import menus, widgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.patches import Circle, FancyArrowPatch
from PyQt5 import QtCore, QtWidgets, QtGui

logger = logging.getLogger(__name__)


class GraphWidget(QtWidgets.QWidget):
    """ """

    def __init__(self, mainwindow):
        """
        Constructor
        """
        super().__init__()

        self.mainwindow = mainwindow
        self.signals = mainwindow.signals
        self.project = mainwindow.project

        self.construct_widget()
        self.connect_signals()

        self.last_node_clicked = []
        self.last_edge_clicked = None

        self.coord = ()

        self.nodes = []
        self.edges = []

        self.init_edge_thread()
        self.init_node_thread()
        # self.init_delete_thread()

        self.signals.name_changed.connect(self.change_node_name)
        self.signals.edge_added.connect(self.plot_edge)
        self.signals.node_added.connect(self.plot_node)

        self.signals.node_removed.connect(self.remove_node)

        self.signals.edge_removed.connect(self.remove_edgepatch)
        self.signals.edge_reversed.connect(self.remove_edgepatch)
        self.signals.edge_reversed.connect(lambda parent, child: self.plot_edge(child, parent))

        self.signals.node_about_to_be_added.connect(self.start_add_node)
        self.signals.edge_about_to_be_added.connect(self.start_add_edge)
        # self.signals.on_click_delete_about_to_happen.connect(self.start_delete_node_edge)

        self.signals.set_graph_message.connect(self.message.set_text)
        self.signals.set_graph_message.connect(lambda s: self.canvas.draw_idle())

        self.signals.nodeview_row_changed.connect(self.select_node)
        self.signals.edgeview_row_changed.connect(self.select_edge)

        self.signals.stop_running_click_threads.connect(self.stop_threads)

        self.node_radius = 0.05

        self.picked = False

    def construct_widget(self):
        """
        Constructs the widget.
        """

        # Create figure
        self.figure = plt.figure(constrained_layout=False)
        self.ax = self.figure.add_axes([0.002, 0.002, 0.998, 0.998])

        # Set background color
        bgcolor = self.palette().color(self.backgroundRole()).name()
        self.figure.patch.set_facecolor(bgcolor)
        # self.figure.tight_layout()

        # self.ax.spines["right"].set_visible(False)
        # self.ax.spines["top"].set_visible(False)
        self.ax.set(aspect=1.0, xticks=[], yticks=[])
        self.ax.axis((0, 1, 0, 1))
        self.message = self.ax.text(0.02, 0.98, s="", ha="left", va="top")

        # Add canvas
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setContentsMargins(0, 0, 0, 0)

        # Buttonbar
        self.add_node_button = QtWidgets.QPushButton("Add node", clicked=self.signals.node_about_to_be_added)
        self.add_edge_button = QtWidgets.QPushButton("Add edge", clicked=self.signals.edge_about_to_be_added)
        self.delete_button = QtWidgets.QPushButton(
            "Delete selected node/edge", clicked=self.delete_selected_object
        )
        self.buttonbar = widgets.HLayout(
            [self.add_node_button, self.add_edge_button, self.delete_button, "stretch"]
        )
        self.setLayout(widgets.VLayout([self.buttonbar, self.canvas, self.toolbar]))

    def connect_signals(self):
        self.canvas.mpl_connect("pick_event", self.onpick)
        self.canvas.mpl_connect("button_release_event", self.onrelease)
        self.canvas.mpl_connect("button_press_event", self.onpress)

    def disconnect_all(self):
        for node in self.nodes:
            node.disconnect()

    def deselect_all(self):
        for node in self.nodes:
            node.deselect()

        for edge in self.edges:
            edge.deselect()

    def onpick(self, event) -> None:

        self.deselect_all()
        self.disconnect_all()

        # We only want a single pick per click
        if self.picked:
            return None

        # On right click
        if event.mouseevent.button == 3:
            self.last_node_clicked.clear()
            self.last_edge_clicked = None

        self.picked = True
        if isinstance(event.artist, Circle):
            # Get circlepatch
            for nodepatch in self.nodes[::-1]:
                if nodepatch.circle == event.artist:
                    self.signals.selected.emit(nodepatch.node.name)
                    # Keep track of the last clicked nodes, needed for drawing edges
                    self.last_node_clicked.append(nodepatch.node.name)

                    # If left click, connect for draging the node to a new position
                    if event.mouseevent.button == 1:
                        nodepatch.connect()
                        # The event-handling for the node is connected *after* the node is clicked.
                        # Therefore, the original click result needs to be emulated.
                        nodepatch.press = nodepatch.circle.center, (nodepatch.node.x, nodepatch.node.y)
                    break

        elif isinstance(event.artist, FancyArrowPatch):
            # Get circlepatch
            for edge in self.edges[::-1]:
                if edge.arrow == event.artist:
                    self.signals.selected.emit((edge.parentnode.node.name, edge.childnode.node.name))
                    self.last_edge_clicked = edge
                    break

    def select_node(self, new, old):
        for nodepatch in self.nodes:
            if nodepatch.node.name == new:
                nodepatch.select()
            if nodepatch.node.name == old:
                nodepatch.deselect()
        self.canvas.draw_idle()

    def select_edge(self, new, old):
        for edgepatch in self.edges:
            if edgepatch.parentnode.node.name == new[0] and edgepatch.childnode.node.name == new[1]:
                edgepatch.select()
            if edgepatch.parentnode.node.name == old[0] and edgepatch.childnode.node.name == old[1]:
                edgepatch.deselect()
        self.canvas.draw_idle()

    def onpress(self, mouseevent) -> None:
        if mouseevent.button == 3:
            return None

        if not self.picked:
            self.deselect_all()
        self.canvas.draw_idle()

    def onrelease(self, mouseevent) -> None:
        if mouseevent.button == 3:
            return None

        self.picked = False
        self.coord = (mouseevent.xdata, mouseevent.ydata)

    def contextMenuEvent(self, event) -> None:
        """
        Creates the context menu for the expert widget
        """
        node = None
        edge = None

        if self.picked and len(self.last_node_clicked) > 0:
            node = self.last_node_clicked[-1]

        elif self.picked and (self.last_edge_clicked is not None):
            edge = self.last_edge_clicked

        menu = menus.GraphContextMenu(self, event, node=node, edge=edge)

    def plot_node(self, crd: tuple, node: Node) -> None:

        nodepatch = NodePatch(
            ax=self.ax, node=node, radius=self.node_radius, facecolor="white", edgecolor="black"
        )
        nodepatch.connect()
        self.nodes.append(nodepatch)

        self.canvas.draw_idle()

    def remove_node(self, name=None) -> None:

        if name is None:
            name = self.last_node_clicked[-1]

        # Remove the patch
        nodepatch = self._get_node(name)
        nodepatch.remove()
        self.nodes.remove(nodepatch)

        # Remove the edge patches
        for edge in self.edges[::-1]:
            if (edge.childnode == nodepatch) or (edge.parentnode == nodepatch):
                edge.remove()
                self.edges.remove(edge)

        self.picked = False

        self.canvas.draw_idle()

    def remove_edgepatch(self, parent: str, child: str) -> None:

        for edge in self.edges:
            if edge.parentnode.node.name == parent and edge.childnode.node.name == child:
                break

        # Remove patch
        edge.remove()
        self.edges.remove(edge)

        # Remove adjacent edges in nodes
        for node in self.nodes:
            for i in reversed(range(len(node.connected_edges))):
                adjacent = node.connected_edges[i][0]
                if edge == adjacent:
                    del node.connected_edges[i]

        self.picked = False

        self.canvas.draw_idle()

    def delete_selected_object(self):
        for nodepatch in self.nodes:
            if nodepatch.selected:
                self.signals.node_about_to_be_removed.emit(nodepatch.node.name)

        for edge in self.edges:
            if edge.selected:
                self.signals.edge_about_to_be_removed.emit(
                    edge.parentnode.node.name, edge.childnode.node.name
                )

    @staticmethod
    def _distance(c1, c2):
        return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5

    def get_nearest_node(self, crd: tuple, search_radius: float = None) -> Union[int, None]:
        dists = {nodepatch.node.name: self._distance(nodepatch.xy, crd) for nodepatch in self.nodes}
        nearest = min(dists, key=dists.get)

        if (search_radius is None) or (dists[nearest] < search_radius):
            return nearest
        else:
            return None

    def wait_for_nodes(self, ncoords, progress_callback=None):
        self.last_node_clicked.clear()
        ndone = 2
        # Collect two coordinates
        while len(self.last_node_clicked) < ndone and self.edge_thread_running:
            time.sleep(0.1)

        if not self.edge_thread_running:
            return None

        if ncoords == 2:
            node1 = self.last_node_clicked[-2]
            node2 = self.last_node_clicked[-1]

            if node1 == node2:
                raise ValueError("Select two different nodes when drawing an edge.")

            self.edge_thread.finished.emit()
            return (node1, node2)

        else:
            raise ValueError()

    def wait_for_node_coordinate(self, progress_callback=None):

        # Collect two coordinateF
        while self.coord is None and self.node_thread_running:
            time.sleep(0.1)

        if not self.node_thread_running:
            return None

        return self.coord

    # def wait_for_delete_coordinate(self, progress_callback=None):

    #     # Collect two coordinates
    #     while self.coord is None and self.delete_thread_running:
    #         time.sleep(0.1)

    #     if not self.delete_thread_running:
    #         return None

    #     return self.coord

    # def delete_coordinate_selected(self):

    #     if self.last_edge_clicked is not None:
    #         edge = self.last_edge_clicked
    #         self.signals.edge_about_to_be_removed.emit(edge.parentnode.node.name, edge.childnode.node.name)
    #         self.delete_thread.finished.emit()

    #     elif len(self.last_node_clicked) > 0:
    #         node = self.last_node_clicked[-1]
    #         self.signals.node_about_to_be_removed.emit(node)
    #         self.delete_thread.finished.emit()

    def init_edge_thread(self):
        # Step 2: Create a QThread object
        self.edge_thread = QtCore.QThread()
        # Step 3: Create a worker object
        self.edge_worker = Worker(self.wait_for_nodes, ncoords=2)
        # Step 4: Move worker to the thread
        self.edge_worker.moveToThread(self.edge_thread)
        # Step 5: Connect signals and slots
        self.edge_thread.started.connect(self.edge_worker.run)
        # self.edge_thread.started.connect(lambda: logger.info("Started edge thread"))
        # self.edge_worker.finished.connect(lambda: logger.info("Finished edge thread"))
        # self.edge_worker.finished.connect(self.edge_thread.quit)
        self.edge_worker.result.connect(self.signals.edge_nodes_selected)
        # self.edge_worker.result.connect(lambda: self.edge_worker.finished.emit())
        self.edge_worker.finished.connect(lambda: self.signals.set_graph_message.emit(""))

    def init_node_thread(self):
        # Step 2: Create a QThread object
        self.node_thread = QtCore.QThread()
        # Step 3: Create a worker object
        self.node_worker = Worker(self.wait_for_node_coordinate)
        # Step 4: Move worker to the thread
        self.node_worker.moveToThread(self.node_thread)
        # Step 5: Connect signals and slots
        self.node_thread.started.connect(self.node_worker.run)
        # self.node_thread.started.connect(lambda: logger.info("Started node thread"))
        # self.node_worker.finished.connect(lambda: logger.info("Finished node thread"))
        # self.node_worker.finished.connect(self.node_thread.quit)
        self.node_worker.result.connect(self.signals.node_coordinate_selected)
        # self.node_worker.result.connect(lambda: self.node_worker.finished.emit())
        self.node_worker.finished.connect(lambda: self.signals.set_graph_message.emit(""))

    # def init_delete_thread(self):
    #     # Step 2: Create a QThread object
    #     self.delete_thread = QtCore.QThread()
    #     # Step 3: Create a worker object
    #     self.delete_worker = Worker(self.wait_for_delete_coordinate)
    #     # Step 4: Move worker to the thread
    #     self.delete_worker.moveToThread(self.delete_thread)
    #     # Step 5: Connect signals and slots
    #     self.delete_thread.started.connect(self.delete_worker.run)
    #     # self.delete_thread.started.connect(lambda: logger.info("Started delete thread"))
    #     # self.delete_worker.finished.connect(lambda: print("Finished delete thread"))
    #     # self.delete_worker.finished.connect(self.delete_thread.quit)
    #     self.delete_worker.result.connect(self.delete_coordinate_selected)
    #     # self.delete_worker.result.connect(lambda: self.delete_worker.finished.emit())
    #     self.delete_worker.finished.connect(lambda: self.signals.set_graph_message.emit(""))

    def stop_threads(self):

        if self.node_thread.isRunning():
            self.node_thread_running = False
            self.node_thread.quit()

        # if self.delete_thread.isRunning():
        #     self.delete_thread_running = False
        #     self.delete_thread.quit()

        if self.edge_thread.isRunning():
            self.edge_thread_running = False
            self.edge_thread.quit()

    def start_add_edge(self):
        # Deselect all nodes
        self.deselect_all()
        self.disconnect_all()
        # Make sure the other click threads are stopped
        self.signals.stop_running_click_threads.emit()
        self.signals.set_graph_message.emit("Select two nodes to draw an edge")
        self.canvas.draw_idle()
        self.edge_thread_running = True
        time.sleep(0.001)
        self.edge_thread.start()

    def start_add_node(self):
        self.coord = None
        self.signals.stop_running_click_threads.emit()
        self.signals.set_graph_message.emit("Select the node position")
        self.canvas.draw_idle()
        self.node_thread_running = True
        self.node_thread.start()

    # def start_delete_node_edge(self):
    #     self.coord = None
    #     self.last_edge_clicked = None
    #     self.last_node_clicked.clear()
    #     self.signals.stop_running_click_threads.emit()
    #     self.signals.set_graph_message.emit("Select the node or edge to delete")
    #     self.canvas.draw_idle()
    #     self.delete_thread_running = True
    #     self.delete_thread.start()

    def _get_node(self, name):
        nodes = [nodepatch for nodepatch in self.nodes if nodepatch.node.name == name]
        assert len(nodes) == 1
        return nodes[0]

    def plot_edge(self, parent, child):

        ntail = self._get_node(parent)
        nhead = self._get_node(child)

        edgepatch = EdgePatch(self.ax, parentnode=ntail, childnode=nhead)
        self.edges.append(edgepatch)

        ntail.connected_edges.append([edgepatch, "tail"])
        nhead.connected_edges.append([edgepatch, "head"])

        self.canvas.draw_idle()

    def change_node_name(self):

        # Get text object with old name and change name to new name
        for nodepatch in self.nodes:
            nodepatch.update_label()
        self.canvas.draw_idle()

    def change_edge_labels(self):
        for edgepatch in self.edges:
            edgepatch.update_label()
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


class EdgePatch:
    def __init__(self, ax, parentnode, childnode):
        self.childnode = childnode
        self.parentnode = parentnode

        self.xtail, self.ytail = parentnode.node.x, parentnode.node.y
        self.xhead, self.yhead = childnode.node.x, childnode.node.y
        self.ax = ax

        chead, ctail, textx, texty, textrot = self.get_positions()

        self.arrow = FancyArrowPatch(
            ctail, chead, mutation_scale=15, color="k", shrinkA=0, shrinkB=0, picker=50, linewidth=0
        )
        self.ax.add_patch(self.arrow)

        self.label = self.ax.text(textx, texty, "", ha="center", va="center", rotation=textrot)

        self.selected = False

    def remove(self):
        self.arrow.remove()
        self.label.remove()

    def update_position(self, ctail=None, chead=None):
        # Check which of the connecting nodes had changed
        if ctail is not None:
            self.xtail, self.ytail = ctail
        if chead is not None:
            self.xhead, self.yhead = chead

        # Get the new radius offsets
        chead, ctail, textx, texty, textrot = self.get_positions()
        # Set the new arrow positions
        self.arrow.set_positions(ctail, chead)

        # Change position of text
        self.label.set_x(textx)
        self.label.set_y(texty)
        self.label.set_rotation(textrot)

    def get_positions(self, radius=0.05):
        alpha = np.arctan2((self.yhead - self.ytail), (self.xhead - self.xtail))
        dx = np.cos(alpha) * radius
        dy = np.sin(alpha) * radius

        chead = (self.xhead - dx, self.yhead - dy)
        ctail = (self.xtail + dx, self.ytail + dy)

        alpha = ((alpha + 0.5 * np.pi) % np.pi) - 0.5 * np.pi
        textx = (self.xtail + self.xhead) * 0.5 + np.cos(alpha + 0.5 * np.pi) * 0.015
        texty = (self.ytail + self.yhead) * 0.5 + np.sin(alpha + 0.5 * np.pi) * 0.015
        textrot = np.degrees(alpha)

        return chead, ctail, textx, texty, textrot

    def toggle_selection(self):
        if self.selected:
            self.deselect()
        else:
            self.select()

    def select(self):
        self.selected = True
        self.arrow.set_facecolor("red")

    def deselect(self):
        self.selected = False
        self.arrow.set_facecolor("black")

    def update_label(self):
        # Get the rank correlation string from the edge tabel
        iparent = self.childnode.node.parent_index(self.parentnode.node.name)
        string = self.childnode.node.edges[iparent].string
        self.label.set_text(string.replace("r_", "r$_{") + "}$")


class NodePatch:
    def __init__(self, ax, node, radius, facecolor="white", edgecolor="black"):
        self.ax = ax
        self.node = node
        self.circle = Circle(
            xy=(node.x, node.y), radius=radius, facecolor=facecolor, edgecolor=edgecolor, picker=True
        )
        self.ax.add_patch(self.circle)
        self.label = self.ax.text(node.x, node.y, "", ha="center", va="center")
        self.update_label()

        self.press = None
        self.selected = False
        self.events_connected = False

        self.connected_edges = []

    def remove(self):
        self.circle.remove()
        self.label.remove()

    def update_label(self):
        self.label.set_text(self.node.name)

    def toggle_selection(self):
        if self.selected:
            self.deselect()
        else:
            self.select()

    def select(self):
        self.selected = True
        self.circle.set_edgecolor("red")
        self.circle.set_linewidth(1.5)

    def deselect(self):
        self.selected = False
        self.circle.set_edgecolor("black")
        self.circle.set_linewidth(1)

    def change_position(self, xynew):
        # Change position of circle
        self.circle.set_center(xynew)
        self.node.x, self.node.y = xynew

        # Change position of egdes
        for edge, end in self.connected_edges:
            edge.update_position(
                ctail=xynew if end == "tail" else None, chead=xynew if end == "head" else None
            )

        # Change position of text
        self.label.set_x(xynew[0])
        self.label.set_y(xynew[1])

    def connect(self):
        """Connect to all the events we need."""
        if not self.events_connected:
            self.events_connected = True
            self.cidpress = self.ax.figure.canvas.mpl_connect("button_press_event", self.on_press)
            self.cidrelease = self.ax.figure.canvas.mpl_connect("button_release_event", self.on_release)
            self.cidmotion = self.ax.figure.canvas.mpl_connect("motion_notify_event", self.on_motion)

    def on_press(self, event):
        """Check whether mouse is over us; if so, store some data."""
        if event.inaxes != self.ax:
            return
        contains = self.circle.contains_point((event.x, event.y))
        if not contains:
            return
        self.press = self.circle.center, (event.xdata, event.ydata)

    def on_motion(self, event):
        """Move the rectangle if the mouse is over us."""
        if self.press is None or event.inaxes != self.ax:
            return
        (x0, y0), (xpress, ypress) = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.change_position((x0 + dx, y0 + dy))

        self.ax.figure.canvas.draw_idle()

    def on_release(self, event):
        """Clear button press information."""
        self.press = None
        self.ax.figure.canvas.draw()

    def disconnect(self):
        """Disconnect all callbacks."""
        if self.events_connected:
            self.events_connected = False
            self.ax.figure.canvas.mpl_disconnect(self.cidpress)
            self.ax.figure.canvas.mpl_disconnect(self.cidrelease)
            self.ax.figure.canvas.mpl_disconnect(self.cidmotion)
