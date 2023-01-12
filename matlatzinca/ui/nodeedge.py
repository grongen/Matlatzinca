from matlatzinca.ui import widgets, menus
from PyQt5 import Qt, QtCore, QtWidgets


class NodeEdgeWidget(QtWidgets.QWidget):
    """ """

    def __init__(self, mainwindow):
        """
        Constructor
        """
        super().__init__()

        self.mainwindow = mainwindow
        self.project = mainwindow.project
        self.signals = mainwindow.signals

        self.signals.lists_about_to_change.connect(self._emit_layout_about_to_be_changed)
        self.signals.lists_changed.connect(self._emit_layout_changed)

        self.construct_widget()

        self.signals.selected.connect(self.set_selection)
        self.nodeview.selectionModel().currentRowChanged.connect(self.emit_node_selected)
        self.edgeview.selectionModel().currentRowChanged.connect(self.emit_edge_selected)

    def set_selection(self, obj):

        # If selection is node
        if isinstance(obj, str):
            # Get the selected row
            row, node = [(i, node) for i, node in enumerate(self.nodemodel.modellist) if node.name == obj][0]
            # Get the model index matching the row
            index = self.nodemodel.index(row, 0)
            # If the row is already selected, the currentRowChanged signal is not triggered, so
            # emit a signal from here
            if self.nodeview.currentIndex().row() == row:
                self.signals.nodeview_row_changed.emit(node.name, "")
            # self.nodeview.setFocus()
            self.nodeview.selectionModel().setCurrentIndex(index, QtCore.QItemSelectionModel.SelectCurrent)
            self.edgeview.selectionModel().clearSelection()

        # If selection is edge
        if isinstance(obj, tuple):
            parent, child = obj
            row, edge = [
                (i, edge)
                for i, edge in enumerate(self.edgemodel.modellist)
                if (edge.child == child) and (parent == edge.parent)
            ][0]
            index = self.edgemodel.index(row, 0)
            # If the row is already selected, the currentRowChanged signal is not triggered, so
            # emit a signal from here
            if self.edgeview.currentIndex().row() == row:
                self.signals.edgeview_row_changed.emit((edge.parent, edge.child), ("", ""))
            # self.edgeview.setFocus()
            self.edgeview.selectionModel().setCurrentIndex(
                index, QtCore.QItemSelectionModel.SelectCurrent | QtCore.QItemSelectionModel.Rows
            )
            self.nodeview.selectionModel().clearSelection()

    def construct_widget(self):
        """
        Constructs the widget.
        """

        # Node view
        self.nodeview = NodeTableView(self)
        # self.nodeview.setStyleSheet("selection-background-color: rgb(204, 232, 255); color: rgb(0, 0, 0);")

        self.nodemodel = NodeListModel(self.project, self.mainwindow)
        self.nodeview.setModel(self.nodemodel)

        self.nodeview.horizontalHeader().setStretchLastSection(True)

        # Edge view
        self.edgeview = EdgeTableView(self)
        # self.edgeview.setStyleSheet("selection-background-color: rgb(204, 232, 255); color: rgb(0, 0, 0);")

        self.edgemodel = EdgeListsModel(self.project, self.mainwindow)
        self.edgeview.setModel(self.edgemodel)
        edgeheader = self.edgeview.horizontalHeader()
        for i in range(self.edgemodel.columnCount()):
            edgeheader.setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)
        edgeheader.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)

        self.setLayout(widgets.HLayout([self.nodeview, self.edgeview], [1, 3]))

    def emit_node_selected(self, new, old):
        self.signals.nodeview_row_changed.emit(new.data(), old.data())

    def emit_edge_selected(self, new, old):
        newedge = self.edgemodel.modellist[new.row()]
        newedge_tuple = (newedge.parent, newedge.child)
        oldrow = old.row()
        if oldrow > (len(self.edgemodel.modellist) - 1):
            oldedge_tuple = ("", "")
        else:
            oldedge = self.edgemodel.modellist[old.row()]
            oldedge_tuple = (oldedge.parent, oldedge.child)
        self.signals.edgeview_row_changed.emit(newedge_tuple, oldedge_tuple)

    def _emit_layout_about_to_be_changed(self):
        self.edgemodel.layoutAboutToBeChanged.emit()
        self.nodemodel.layoutAboutToBeChanged.emit()

    def _emit_layout_changed(self):
        self.edgemodel.layoutChanged.emit()
        self.nodemodel.layoutChanged.emit()


class ReorderTableView(QtWidgets.QTableView):
    """QTableView with the ability to make the model move a row with drag & drop"""

    class DropmarkerStyle(QtWidgets.QProxyStyle):
        def drawPrimitive(self, element, option, painter, widget=None):
            """Draw a line across the entire row rather than just the column we're hovering over.
            This may not always work depending on global style - for instance I think it won't
            work on OSX."""
            if element == self.PE_IndicatorItemViewItemDrop and not option.rect.isNull():
                option_new = QtWidgets.QStyleOption(option)
                option_new.rect.setLeft(0)
                if widget:
                    option_new.rect.setRight(widget.width())
                option = option_new
            super().drawPrimitive(element, option, painter, widget)

    def __init__(self, parent):
        super().__init__(parent)

        self.setSelectionBehavior(self.SelectRows)
        self.setSelectionMode(self.SingleSelection)
        self.setDragDropMode(self.InternalMove)
        self.setDragDropOverwriteMode(False)

        self.setStyle(self.DropmarkerStyle())
        self.mainwindow = parent.mainwindow
        self.signals = self.mainwindow.signals

        self.setShowGrid(False)
        self.setAlternatingRowColors(True)
        self.setStyleSheet("QTableView{border: 1px solid " + self.mainwindow.bordercolor + "}")

    def dropEvent(self, event):
        if event.source() is not self or (
            event.dropAction() != QtCore.Qt.MoveAction
            and self.dragDropMode() != QtWidgets.QAbstractItemView.InternalMove
        ):
            super().dropEvent(event)

        selection = self.selectedIndexes()
        from_index = selection[0].row() if selection else -1
        to_index = self.indexAt(event.pos()).row()
        if (
            0 <= from_index < self.model().rowCount()
            and 0 <= to_index < self.model().rowCount()
            and from_index != to_index
        ):

            return from_index, to_index


class NodeTableView(ReorderTableView):
    def __init__(self, parent):
        super().__init__(parent)

    def contextMenuEvent(self, event):
        """
        Creates the context menu for the expert widget
        """
        menus.NodeViewContextMenu(self, event)

    def delete_node(self, nodename):
        self.signals.node_about_to_be_removed.emit(nodename)

    def dropEvent(self, event):
        res = super().dropEvent(event)
        if res is not None:
            from_index, to_index = res
            self.signals.node_order_about_to_change.emit(from_index, to_index)


class EdgeTableView(ReorderTableView):
    """QTableView with the ability to make the model move a row with drag & drop"""

    def __init__(self, parent):
        super().__init__(parent)
        self.signals = parent.signals

    def contextMenuEvent(self, event):
        """
        Creates the context menu for the expert widget
        """
        menus.EdgeViewContextMenu(self, self.signals, event)

    def dropEvent(self, event):
        res = super().dropEvent(event)
        if res is not None:
            from_index, to_index = res
            self.signals.parent_order_about_to_change.emit(from_index, to_index)


def strformat(item):
    """
    String formatter

    Parameters
    ----------
    item : str or int or float
        Item to be converted to formatted string

    Returns
    -------
    str
        Formatted string
    """
    return item if isinstance(item, str) else "{:.3g}".format(item).replace("nan", "")


class ListsModel(QtCore.QAbstractTableModel):
    """
    Class to populate a table view with a pandas dataframe
    """

    def __init__(self, modellist, keys, labels):
        QtCore.QAbstractTableModel.__init__(self)
        self.modellist = modellist
        self.keys = keys
        self.labels = labels
        self.leftalign = []

        # Check label dimensions
        if len(labels) != len(keys):
            raise ValueError(f"Requires an equal number of keys ({len(keys)}) and labels ({len(labels)}).")

    def rowCount(self, parent=None):
        """Returns number of rows in table."""
        return len(self.modellist)

    def columnCount(self, parent=None):
        """Returns number of columns in table."""
        return len(self.labels)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        """
        Method to set the data to a cell. Sets value, alignment and color

        Parameters
        ----------
        index : int
            Index object with row and column
        role : Qt.DisplayRole, optional
            Property of data to display, by default QtCore.Qt.DisplayRole

        Returns
        -------
        QtCore.QVariant
            Property of data displayed
        """
        if not index.isValid():
            return None

        # Get number from lists
        if role == QtCore.Qt.DisplayRole:
            # Get row
            row = index.row()
            col = index.column()

            # If within length of list
            if row < len(self.modellist):
                # Get item
                item = getattr(self.modellist[row], self.keys[col])
                # Return directly if string
                if isinstance(item, str):
                    return item
                # Else convert to string
                else:
                    return strformat(item)

        # Alignment
        elif role == QtCore.Qt.TextAlignmentRole:
            if index.column() in self.leftalign:
                return QtCore.QVariant(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            else:
                return QtCore.QVariant(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def headerData(self, rowcol, orientation, role):
        """
        Method to get header (index and columns) data
        """
        # Show column labels
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return str(self.labels[rowcol])

        if orientation == QtCore.Qt.Vertical and role == QtCore.Qt.DisplayRole:
            return str(rowcol + 1)


class NodeListModel(ListsModel):
    """
    Class to populate a table view with several lists related to the items
    """

    def __init__(self, project, mainwindow):
        """
        Constructor

        Parameters
        ----------
        parentwidget : Table widget
            Widget of which this table model is part.
        """
        self.project = project
        self.mainwindow = mainwindow
        self.signals = mainwindow.signals

        super().__init__(modellist=self.project.bn.nodes, keys=["name"], labels=["Nodes\n"])

        self.editable = [True]

    def setData(self, index, value, role=Qt.Qt.EditRole):
        """
        Method to set data to cell. Changes the value in the list or array
        """
        if not index.isValid():
            return False

        col = index.column()
        row = index.row()

        oldvalue = getattr(self.modellist[row], self.keys[col])
        if oldvalue == value:
            return False

        self.signals.name_about_to_change.emit(oldvalue, value)

        return True

    def flags(self, index) -> QtCore.Qt.ItemFlags:
        """
        Returns flags (properties) for a certain cell, based on the index

        Parameters
        ----------
        index : ModelIndex
            index of the cell

        Returns
        -------
        flags
            Properties of the cell
        """
        col = index.column()

        fl = QtCore.Qt.NoItemFlags
        if not index.isValid():
            # pass
            fl |= QtCore.Qt.ItemIsDropEnabled
        else:
            fl = QtCore.Qt.ItemIsSelectable
            fl |= Qt.Qt.ItemIsEnabled | QtCore.Qt.ItemIsDragEnabled
            if self.editable[col]:
                fl |= Qt.Qt.ItemIsEditable
        return fl

    def supportedDropActions(self) -> bool:
        return QtCore.Qt.MoveAction | QtCore.Qt.CopyAction


class EdgeListsModel(ListsModel):
    """
    Class to populate a table view with several lists related to the items
    """

    def __init__(self, project, mainwindow):
        """
        Constructor

        Parameters
        ----------
        parentwidget : Table widget
            Widget of which this table model is part.
        """
        self.project = project
        self.mainwindow = mainwindow
        self.signals = mainwindow.signals

        super().__init__(
            modellist=self.project.bn.edgelist,
            labels=[
                "Edge",
                "Conditional\nrank corr.",
                "Non-conditional\nrank corr.",
                "Range non-cond.\nrank corr.",
            ],
            keys=["direction_string", "cond_rank_corr", "rank_corr", "rank_corr_range"],
        )

        self.editable = [False, True, True, False]

    def setData(self, index, value: str, role=Qt.Qt.EditRole) -> bool:
        """
        Method to set data to cell. Changes the value in the list or array
        """
        if not index.isValid():
            return False

        col = index.column()
        row = index.row()

        # If empty value is given
        if value == "":
            return False

        if self.keys[col] == "cond_rank_corr":
            self.signals.correlation_about_to_change.emit(
                self.modellist[row].parent, self.modellist[row].child, value
            )
        elif self.keys[col] == "rank_corr":
            self.signals.observed_correlation_about_to_change.emit(
                self.modellist[row].parent, self.modellist[row].child, value
            )
        else:
            raise KeyError(self.keys[col])

        return True

    def data(self, index, role=QtCore.Qt.DisplayRole):

        result = super().data(index, role)

        # Get number from lists
        if role == QtCore.Qt.DisplayRole:
            col, row = index.column(), index.row()
            if self.keys[col] == "cond_rank_corr":
                result = self.modellist[row].cond_rstring + " = " + result
            elif self.keys[col] == "rank_corr":
                result = self.modellist[row].uncond_rstring + " = " + result

        return result

    def flags(self, index) -> QtCore.Qt.ItemFlags:
        """
        Returns flags (properties) for a certain cell, based on the index

        Parameters
        ----------
        index : ModelIndex
            index of the cell

        Returns
        -------
        flags
            Properties of the cell
        """
        col = index.column()

        fl = QtCore.Qt.NoItemFlags
        if not index.isValid():
            fl |= QtCore.Qt.ItemIsDropEnabled
        else:
            fl = QtCore.Qt.ItemIsSelectable
            fl |= Qt.Qt.ItemIsEnabled | QtCore.Qt.ItemIsDragEnabled
            if self.editable[col]:
                fl |= Qt.Qt.ItemIsEditable
        return fl

    def supportedDropActions(self) -> bool:
        return QtCore.Qt.MoveAction | QtCore.Qt.CopyAction
