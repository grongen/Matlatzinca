import logging
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from matlatzinca.core.bn import BayesianNetwork
from matlatzinca.io import get_table_text
from matlatzinca.ui.dialogs import NotificationDialog
from PyQt5 import QtWidgets

logger = logging.getLogger(__name__)


class Project:
    def __init__(self, mainwindow) -> None:
        self.mainwindow = mainwindow
        self.bn = BayesianNetwork()

        self.signals = self.mainwindow.signals

        self.signals.node_coordinate_selected.connect(self.add_node)
        self.signals.node_about_to_be_removed.connect(self.remove_node)

        self.signals.edge_nodes_selected.connect(self.add_edge)
        self.signals.edge_about_to_be_removed.connect(self.remove_edge)
        self.signals.edge_about_to_be_reversed.connect(self.reverse_edge)

        # self.signals.delete_coordinate_selected.connect(self.add_node)

        self.signals.observed_correlation_about_to_change.connect(self.change_observed_correlation)
        self.signals.correlation_about_to_change.connect(self.change_edge_correlation)
        self.signals.name_about_to_change.connect(self.change_node_name)

        # self.signals.sampling_order_about_to_change.connect(self.change_sampling_order)
        self.signals.node_order_about_to_change.connect(self.change_node_order)
        self.signals.parent_order_about_to_change.connect(self.change_parent_order)

    def update_bn(self) -> None:
        """Updates the BN model by calculating the correlation matrix, correlation bounds, and edges"""
        if len(self.bn.nodes) == 0:
            self.bn.R = np.empty((0, 0), dtype=np.float64)
        else:
            logger.info("Calculating correlation matrix and bounds.")
            # self.bn.check_sampling_order()
            self.bn.calculate_correlation_matrix()
            self.bn.calculate_correlation_bounds()
            self.bn.create_edge_overview()

    def add_node(self, crd: Union[Tuple[float, float], None]) -> None:
        """Add node at specific coordinate"""
        if crd is None:
            return None

        self.signals.lists_about_to_change.emit()
        # Add node to BN
        name = self.bn._get_unused_name()
        self.bn.add_node(name=name, parents=[], rank_corrs=[])
        # Add coordinates to Node
        self.update_bn()

        logger.info(f'Node "{name}" added.')
        self.signals.lists_changed.emit()

        # Get the node and add coordinates
        node = self.bn._get_node_by_name(name)
        node.x, node.y = crd

        # Emit the signal that will update the plots
        self.signals.node_added.emit(crd, node)

    def remove_node(self, name: str) -> None:
        """Remove node by name"""
        self.signals.lists_about_to_change.emit()
        self.bn.remove_node(name)
        logger.info(f'Node "{name}" removed.')
        self.update_bn()
        self.signals.node_removed.emit(name)
        self.signals.lists_changed.emit()

    def remove_edge(self, parent: str, child: str) -> None:
        """Remove edge by parent and child name"""
        self.signals.lists_about_to_change.emit()
        self.bn.remove_edge(parent, child)
        logger.info(f'Edge from "{parent}" to "{child}" removed.')
        self.update_bn()
        self.signals.edge_removed.emit(parent, child)
        self.signals.lists_changed.emit()

    def add_edge(self, edge: Union[Tuple[str, str], None]) -> None:
        """Add edge from parent and child string"""
        if edge is None:
            return None

        self.signals.lists_about_to_change.emit()
        succes = self.bn.add_edge(*edge)
        if succes:
            logger.info(f'Edge from "{edge[0]}" to "{edge[1]}" added.')
            self.update_bn()
            # Emit the signal that will update the plots
            self.signals.edge_added.emit(*edge)

        else:
            NotificationDialog("Edge was not added, as the BN would no longer be a DAG.")
        self.signals.lists_changed.emit()

    def reverse_edge(self, parent: str, child: str) -> None:
        """Reverse an existing edge (and recalculate conditional rank correlations)"""

        self.signals.lists_about_to_change.emit()
        succes = self.bn.reverse_edge(parent, child)
        if succes:
            logger.info(f'Edge from "{parent}" to "{child}" reversed.')
            self.update_bn()
            # Emit the signal that will update the plots
            self.signals.edge_reversed.emit(parent, child)

        else:
            NotificationDialog("Edge was not reversed, as the BN would no longer be a DAG.")
        self.signals.lists_changed.emit()

    def change_observed_correlation(self, parent: str, child: str, corr: str) -> None:
        """Change (non-conditional) correlation coefficient"""

        node = self.bn._get_node_by_name(child)
        iparent = node.parent_index(parent)
        oldvalue = node.edges[iparent].rank_corr

        node.edges[iparent].rank_corr = corr
        corr = float(corr)

        # Check if the correlation is within bounds
        edge = node.edges[iparent]
        if corr < edge.rank_corr_bounds[0] or corr > edge.rank_corr_bounds[-1]:
            node.edges[iparent].rank_corr = oldvalue
            raise ValueError("Choose a value in between {:.4g} and {:.4g}".format(*edge.rank_corr_bounds))

        # Calculate the conditional correlation from the observed correlation
        cond_corr = self.bn.calculate_conditional_correlation(parent, child, observed=corr)

        # Change the conditional correlation
        self.change_edge_correlation(parent, child, corr=cond_corr)

    def change_edge_correlation(self, parent: str, child: str, corr: str) -> None:
        """Change the edge correlation"""

        node = self.bn._get_node_by_name(child)
        iparent = node.parent_index(parent)
        node.edges[iparent].cond_rank_corr = corr

        logger.info(f'Conditional edge ("{parent}" to "{child}") correlation changed to {corr}.')
        self.signals.lists_about_to_change.emit()
        self.update_bn()
        self.signals.correlation_changed.emit()
        self.signals.lists_changed.emit()

    def change_node_name(self, oldname: str, newname: str) -> None:
        """Change a node name"""

        if newname == "":
            return None
            # raise ValueError("Name cannot be empty")

        self.signals.lists_about_to_change.emit()

        self.bn.change_node_name(oldname, newname)

        logger.info(f'Node "{oldname}" was renamed to "{newname}".')

        self.signals.name_changed.emit()

        # self.bn.create_edge_overview()

        self.signals.lists_changed.emit()

    def change_node_order(self, source_pos: int, target_pos: int) -> None:
        """Change the node order"""

        self.signals.lists_about_to_change.emit()

        # Change the node order
        current_order = self.bn._node_names
        self.bn.nodes.insert(target_pos, self.bn.nodes.pop(source_pos))
        requested_order = self.bn._node_names

        # Reorder the nodes
        logger.info(f"Node order changed to to ({', '.join(requested_order)})")

        self.update_bn()

        self.signals.node_order_changed.emit()

        self.signals.lists_changed.emit()

    # def change_sampling_order(self, source_pos: int, target_pos: int) -> None:


    #     self.signals.lists_about_to_change.emit()

    #     # Change the node order
    #     self.bn.nodes.insert(target_pos, self.bn.nodes.pop(source_pos))
    #     requested_order = self.bn._node_names

    #     # Reorder the nodes based on the sampling order
    #     self.bn.check_sampling_order()
    #     valid_order = self.bn._node_names

    #     logger.info(f"Sampling order changed to to ({', '.join(valid_order)})")

    #     self.update_bn()

    #     self.signals.sampling_order_changed.emit()

    #     self.signals.lists_changed.emit()

    #     if requested_order != valid_order:
    #         NotificationDialog(
    #             f"The given order ({', '.join(requested_order)}) is not valid sampling order. Therefore, the order is changed to ({', '.join(valid_order)})"
    #         )

    def change_parent_order(self, source_pos: int, target_pos: int) -> None:
        """Change order of parents, as a result of reordering the edges"""
        self.signals.lists_about_to_change.emit()

        # Get edge from source position
        edge = self.bn.edgelist[source_pos]
        # Update order
        self.bn.change_parent_order(source_pos, target_pos)
        # Update BN with correlations
        self.update_bn()

        self.signals.parent_order_changed.emit()

        self.signals.lists_changed.emit()
        # Select the new row with the edge position
        self.signals.selected.emit((edge.parent, edge.child))

    def new(self) -> None:
        """Empty the project"""
        for node in reversed(self.bn.nodes):
            self.signals.node_about_to_be_removed.emit(node.name)

    def open(self, *args, fname: Path = None) -> None:
        """
        Method that loads project file and builds gui
        """

        if fname is None:
            fname = self.get_project_file()
            if fname == "":
                return None
            else:
                fname = Path(fname)

        # Clear current project
        self.new()

        # Open project
        self.mainwindow.setCursorWait()

        # Load from file
        self.signals.lists_about_to_change.emit()
        loaded_bn = BayesianNetwork.parse_file(fname)
        self.bn.nodes.extend(loaded_bn.nodes)
        self.update_bn()

        # Add nodes to UI
        for node in self.bn.nodes:
            self.signals.node_added.emit((np.random.rand(1)[0], np.random.rand(1)[0]), node)

        # Add edged to UI
        for node in self.bn.nodes:
            for edge in node.edges:
                self.signals.edge_added.emit(edge.parent, edge.child)

        self.signals.lists_changed.emit()

        # save current dir
        self.mainwindow.appsettings.setValue("currentdir", str(fname.parent))
        self.mainwindow.update_projectname(fname.name)
        self.signals.set_window_modified.emit(False)
        self.mainwindow.setCursorNormal()

    def save_as(self, fname=None) -> None:
        """
        Asks the user for the file name to save the project to.
        If the file name is passed, the user is not asked the file
        and the project is saved directly.

        Parameters
        ----------
        fname : str, optional
            File name, by default None
        """

        # if fname is None:
        # Set open file dialog settings
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        options |= QtWidgets.QFileDialog.DontConfirmOverwrite

        # Set current dir
        currentdir = self.mainwindow.appsettings.value("currentdir", ".", type=str)

        # Open dialog to select file
        fname, ext = QtWidgets.QFileDialog.getSaveFileName(
            self.mainwindow, "Matlatzinca - Save project", currentdir, "JSON (*.json)", options=options
        )

        if fname == "":
            return None

        # Add extension if not given in file
        for end in [".json"]:
            if end in ext and not fname.endswith(end):
                fname += end

        # Convert to path
        fname = Path(fname)

        if fname.exists():
            reply = QtWidgets.QMessageBox.warning(
                self.mainwindow,
                "Matlatzinca - Save project",
                f"{fname.name} already exists.\nDo you want to overwrite it?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if reply == QtWidgets.QMessageBox.No:
                return None

        # Save files
        self.mainwindow.update_projectname(fname.name)

        self.bn.to_json(fname)
        self.signals.set_window_modified.emit(False)

    def save(self) -> None:
        """
        Save current project, checks if the file name is already known.
        If not, the save as function is called and the user is asked to
        pick a file.
        """

        # If no current project, save as...
        currentproject = self.mainwindow.appsettings.value("currentproject", "", type=str)
        if currentproject == "":
            self.save_project_as()

        else:
            # Get current directory
            currentdir = Path(self.mainwindow.appsettings.value("currentdir", ".", type=str))

            # Else, save files directly
            self.bn.to_json(currentdir / currentproject)
            self.signals.set_window_modified.emit(False)

    def get_project_file(self) -> str:
        """
        Opens a dialog to select a project file to open.
        """
        # Set open file dialog settings
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog

        # Set current dir
        currentdir = self.mainwindow.appsettings.value("currentdir", ".", type=str)

        # Open dialog to select file
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.mainwindow,
            "Matlatzinca - Open project",
            currentdir,
            "JSON (*.json)",
            options=options,
        )

        return fname

    def export(self, what: str, how: str) -> None:
        """Exports correlation matrix or tables to csv or clipboard

        Args:
            what (str): correlation_matrix, nodes, or edges
            how (str): csv or clipboard
        """

        if not what in ["correlation_matrix", "nodes", "edges"]:
            raise KeyError(
                f'Choose one of "{", ".join(["correlation_matrix", "nodes", "edges"])}" for "what"'
            )

        if not how in ["csv", "clipboard"]:
            raise KeyError(f'Choose one of "{", ".join(["csv" , "clipboard"])}" for "how"')

        sep = "\t" if how == "clipboard" else ","

        # Gather data
        if what == "correlation_matrix":
            data = [self.bn._node_names] + self.bn.R[:, :].astype(str).tolist()
            data = [[name] + row for row, name in zip(data, [""] + self.bn._node_names)]
            text = "\n".join([sep.join(row) for row in data])

        elif what == "nodes":
            text = get_table_text(model=self.mainwindow.nodes_edges_widget.nodemodel, newline_replace=" ")

        elif what == "edges":
            text = get_table_text(model=self.mainwindow.nodes_edges_widget.edgemodel, newline_replace=" ")

        # Export data
        if how == "csv":
            # Set open file dialog settings
            options = QtWidgets.QFileDialog.Options() | QtWidgets.QFileDialog.DontUseNativeDialog

            # Set current dir
            currentdir = self.mainwindow.appsettings.value("currentdir", ".", type=str)

            # Open dialog to select file
            fname, ext = QtWidgets.QFileDialog.getSaveFileName(
                self.mainwindow, "BN Drawing - Save data", currentdir, "CSV (*.csv)", options=options
            )

            if fname == "":
                return None
            elif ".csv" in ext and not fname.endswith(".csv"):
                fname += ".csv"

            # Convert to path
            fname = Path(fname)
            with fname.open("w", encoding="utf-8", errors=" ") as f:
                f.write(text)

        elif how == "clipboard":
            QtWidgets.qApp.clipboard().setText(text)
