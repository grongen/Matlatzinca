from PyQt5 import Qt, QtCore, QtGui, QtWidgets


class GraphContextMenu(QtWidgets.QMenu):
    def __init__(self, graph_widget, event, node=None, edge=None):
        super().__init__(graph_widget)
        # Get position

        # rownum = expertswidget.table.currentIndex().row()

        # self.experts = expertswidget.project.experts
        self.graph_widget = graph_widget
        self.signals = graph_widget.mainwindow.signals
        # self.tableheader = self.expertswidget.table.horizontalHeader()

        self.action_dict = {}

        self.construct(node=node, edge=edge)

        self.perform_action(event, node=node, edge=edge)

    def construct(self, node=None, edge=None):

        # Add add actions
        if node is None and edge is None:
            self.action_dict["add_node"] = self.addAction("Add node")
            self.action_dict["add_edge"] = self.addAction("Add edge")
            self.action_dict["remove_node_edge"] = self.addAction("Remove node or edge")

        # Add remove actions
        if node is not None:
            self.action_dict["remove_node"] = self.addAction(f"Remove node {node}")

        if edge is not None:
            edgestr = f"{edge.parentnode.node.name} \u2192 {edge.childnode.node.name}"
            self.action_dict["remove_edge"] = self.addAction(f"Remove edge ({edgestr})")
            self.action_dict["reverse_edge"] = self.addAction(f"Reverse edge ({edgestr})")

    def perform_action(self, event, node, edge):

        # get action
        action = self.exec_(self.graph_widget.mapToGlobal(event.pos()))

        if "add_node" in self.action_dict and action == self.action_dict["add_node"]:
            self.signals.node_about_to_be_added.emit()

        elif "add_edge" in self.action_dict and action == self.action_dict["add_edge"]:
            self.signals.edge_about_to_be_added.emit()

        elif "remove_node_edge" in self.action_dict and action == self.action_dict["remove_node_edge"]:
            self.signals.on_click_delete_about_to_happen.emit()

        elif "remove_node" in self.action_dict and action == self.action_dict["remove_node"]:
            self.signals.node_about_to_be_removed.emit(node)

        elif "remove_edge" in self.action_dict and action == self.action_dict["remove_edge"]:
            self.signals.edge_about_to_be_removed.emit(edge.parentnode.node.name, edge.childnode.node.name)

        elif "reverse_edge" in self.action_dict and action == self.action_dict["reverse_edge"]:
            self.signals.edge_about_to_be_reversed.emit(edge.parentnode.node.name, edge.childnode.node.name)


class NodeViewContextMenu(QtWidgets.QMenu):
    def __init__(self, nodeview, event):
        super().__init__(nodeview)
        # Get position
        rownum = nodeview.currentIndex().row()

        self.nodeview = nodeview
        self.action_dict = {}
        self.construct()
        self.perform_action(event, rownum)

    def construct(self):
        # Add remove actions
        self.action_dict["remove_node"] = self.addAction("Remove Node")

    def perform_action(self, event, rownum):
        # get action
        action = self.exec_(self.nodeview.mapToGlobal(event.pos()))

        if "remove_node" in self.action_dict and action == self.action_dict["remove_node"]:
            self.nodeview.delete_node(self.nodeview.model().modellist[rownum].name)


class EdgeViewContextMenu(QtWidgets.QMenu):
    def __init__(self, edgeview, signals, event):
        super().__init__(edgeview)
        # Get position
        rownum = edgeview.currentIndex().row()

        self.edgeview = edgeview
        self.signals = signals
        self.action_dict = {}
        self.construct()
        self.perform_action(event, rownum)

    def construct(self):
        # Add remove actions
        self.action_dict["remove_edge"] = self.addAction("Remove Edge")
        self.action_dict["reverse_edge"] = self.addAction("Reverse Edge")

    def perform_action(self, event, rownum):
        # get action
        action = self.exec_(self.edgeview.mapToGlobal(event.pos()))

        edge = self.edgeview.model().modellist[rownum]
        if "remove_edge" in self.action_dict and action == self.action_dict["remove_edge"]:
            self.signals.edge_about_to_be_removed.emit(edge.parent, edge.child)

        elif "reverse_edge" in self.action_dict and action == self.action_dict["reverse_edge"]:
            self.signals.edge_about_to_be_reversed.emit(edge.parent, edge.child)
