import math
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

from matlatzinca.core.models import BaseModel, Edge, Node


def corr_string(r: Union[float, None], i: int, j: int, cond: tuple = None, offset: int = 0) -> str:
    """Return a string for the partial correlation

    Parameters
    ----------
    r : Union[float, None]
        Correlation coefficient
    i : int
        Node 1 integer
    j : int
        Node 2 integer
    cond : tuple, optional
        integers of conditional nodes. Defaults to None.
    offset : int, optional
        Offset all integers with a certain value. Defaults to 0.

    Returns
    -------
    str
        Correlation string
    """

    if cond is not None and len(cond) > 0:
        cond_str = "|" + ",".join([f"{c+offset}" for c in cond])
    else:
        cond_str = ""
    if r is not None:
        rstr = f" = {r:.3f}"
    else:
        rstr = ""
    string = "r_" + f"{i+offset},{j+offset}{cond_str}{rstr}"
    return string


def _ranktopearson(R: float) -> float:
    """Transforming rank correlation (R) into Pearson's correlation (r)

    Parameters
    ----------
    R : float
        Rank correlation

    Returns
    -------
    float
        Pearson's correlation
    """

    if abs(R) == 1:
        return R
    else:
        r = 2 * math.sin((math.pi / 6) * R)
        return r


_vranktopearson = np.vectorize(_ranktopearson)


def ranktopearson(R: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """Wrapper function of the non-vectorized and vectorized rank to perason function

    Parameters
    ----------
    R : Union[np.ndarray, float]
        Rank correlation

    Returns
    -------
    Union[np.ndarray, float]
        Pearson's correlation
    """
    if isinstance(R, np.ndarray):
        return _vranktopearson(R)
    else:
        return _ranktopearson(R)


def _pearsontorank(r: float) -> float:
    """Transforming Pearson's correlation (r) into rank correlation (R)

    Parameters
    ----------
    r : float
        Pearson's correlation

    Returns
    -------
    float
        Rank correlation
    """
    if abs(r) == 1:
        return r
    else:
        R = (6 / math.pi) * math.asin(r / 2)
        return R


_vpearsontorank = np.vectorize(_pearsontorank)


def pearsontorank(r: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """Wrapper function of the non-vectorized and vectorized Pearson to rank function

    Parameters
    ----------
    r : Union[np.ndarray, float]
        Pearson's correlation

    Returns
    -------
    Union[np.ndarray, float]
        Rank correlation
    """
    if isinstance(r, np.ndarray):
        return _vpearsontorank(r)
    else:
        return _pearsontorank(r)


class BayesianNetwork(BaseModel):

    # list of nodes
    nodes: List[Node] = []
    # Correlation matrix
    R: np.ndarray = np.empty((0, 0), dtype=float)
    # Bounds of the observed correlation per edge
    # bounds: Dict[Tuple[str, str], Tuple[float, float]] = {}
    # String describing the edge
    # corr_string: Dict[Tuple[str, str], str] = {}
    # Dictionary with partial correlations
    partcorrs: Dict[Tuple, float] = {}
    # List of edges. This is not used for calculations, as the nodes parents
    # already contain all information on the edges
    edgelist: List[Edge] = []

    def add_node(
        self,
        name: str = None,
        parents: List[str] = None,
        rank_corrs: List[float] = None,
    ) -> None:
        """Add node to BN

        Parameters
        ----------
        name : str, optional
            Name of the node. If None, a default name is chosen. Defaults to None.
        parents : List[str], optional
            List of parent nodes (names). Defaults to None.
        rank_corrs : List[float], optional
            Rank correlations corresponding to parent nodes. Defaults to None.
        """
        if name is None:
            name = self._get_unused_name()

        edges = []
        if (parents is not None) and (rank_corrs is not None):
            for parent, corr in zip(parents, rank_corrs):
                edges.append(Edge(parent=parent, child=name, cond_rank_corr=corr))

        newnode = Node(name=name, edges=edges)
        self.nodes.append(newnode)

        # Make sure the edges are ordered by sampling order
        # self.check_sampling_order()

    def remove_node(self, name: str) -> None:
        """Removes a node from the project. The given node is removed.
        Other nodes are checked as well for their parents, and removed
        if necessary, meaning an edge is removed.

        Parameters
        ----------
        name : str
            Name of node to remove.
        """

        inode = self._node_names.index(name)
        node = self.nodes[inode]
        self.nodes.remove(node)

        # Remove edges connected to this node in other nodes
        for node in self.nodes:
            if name in node.parent_names:
                self.remove_edge(parent=name, child=node.name)

        # Make sure the edges are ordered by sampling order
        # self.check_sampling_order()

    def remove_edge(self, parent: str, child: str) -> None:
        """Removes an edge from the BN. All node's parents are checked
        and removed if matching

        Parameters
        ----------
        parent : str
            Edge origin
        child : str
            Edge destination
        """

        inode = self._node_names.index(child)
        node = self.nodes[inode]

        iedge = node.parent_index(parent)
        node.edges.remove(node.edges[iedge])

        # Make sure the edges are ordered by sampling order
        # self.check_sampling_order()

    def _get_unused_name(self) -> str:
        """Generates a node name that is not used already

        Returns:
        -------
        str
            Generated node name
        """
        newid = len(self.nodes) + 1
        newname = f"Node {newid}"
        while newname in self._node_names:
            newid += 1
            newname = f"Node {newid}"
        return newname

    def add_edge(self, parent_name: str, child_name: str, cond_rank_corr: float = 0.0) -> bool:
        """Adds an edge between two nodes to the model.

        Parameters
        ----------
        parent_name : str
            Name of the parent node
        child_name : str
            Name of the child node
        cond_rank_corr : float, optional
            Conditional rank correlation between the two nodes. By default 0.0.

        Returns
        -------
        bool
            Whether succesfull. Returns True when the result is an DAG, else False.

        Raises
        ------
        ValueError
            If the edge already exists
        """

        # Check if edge is already present
        node = self._get_node_by_name(child_name)
        if parent_name in node.parent_names:
            raise ValueError(f'An edge from "{parent_name}" to "{child_name}" already exists.')

        # Create edge
        edge = Edge(parent=parent_name, child=child_name, cond_rank_corr=cond_rank_corr)

        node.edges.append(edge)

        if not self.is_dag:
            self.remove_edge(parent_name, child_name)
            return False

        else:
            # Make sure the edges are ordered by sampling order
            # self.check_sampling_order()

            return True

    def reverse_edge(self, parent_name: str, child_name: str) -> bool:
        """Reverse an existing edge in the BN. This is done by removing the old edge,
        adding the reversed one, checking if it is still a DAG, and if not add the
        old one again.

        Note that the observed correlations are not maintained when reversing edges.

        Parameters
        ----------
        parent_name : str
            Name of the parent node, which becomes the child node
        child_name : str
            Name of the child node, which becomes the parent node

        Returns
        -------
        bool
            If reversing was successfull (BN still a DAG)
        """

        # Get the current rank correlation
        node = self._get_node_by_name(child_name)
        cond_rank_corr = node.edges[node.parent_index(parent_name)].cond_rank_corr

        # Remove the current edge
        self.remove_edge(parent_name, child_name)

        # Add the reversed edge, and check if still DAG
        success = self.add_edge(parent_name=child_name, child_name=parent_name)
        if success:
            return True

        # If not successfull (no DAG), put the old edge back again
        else:
            self.add_edge(
                parent_name=parent_name,
                child_name=child_name,
                cond_rank_corr=cond_rank_corr,
            )
            return False

    @property
    def _node_names(self) -> list:
        """Returns a list of node names"""
        return [node.name for node in self.nodes]

    def _get_node_by_name(self, name: str) -> Node:
        """Get a node object by its name"""
        return self.nodes[self._node_names.index(name)]

    def change_node_name(self, oldname: str, newname: str) -> None:
        """Change node name"""

        # Create a list of other names
        names = self._node_names
        inode = names.index(oldname)
        names.remove(names[inode])

        # Check if newname is already in use by a different node
        if newname in names:
            raise ValueError(f'Name "{newname}" is already in use.')

        # Rename nodes
        node = self._get_node_by_name(oldname)
        node.name = newname
        for edge in node.edges:
            edge.child = newname

        for node in self.nodes:
            # Change name in parents
            if oldname in node.parent_names:
                iparent = node.parent_index(oldname)
                node.edges[iparent].parent = newname

        self.create_edge_overview()

    def create_edge_overview(self) -> None:
        """Creates an internal overview of the edges"""
        del self.edgelist[:]
        for node in self.nodes:
            self.edgelist.extend(node.edges)

    def get_valid_sampling_order(self) -> List[int]:
        """Construct a valid sampling order. This means that the node with no
        parents will be the first in the sampling order (SO) and so forth.

        Returns
        -------
        List[int]
            indices with the sampling order
        """

        nnodes = len(self.nodes)
        names = self._node_names

        # Add disconnected nodes to the end of the list
        disconnected = set(names)
        for node in self.nodes:
            # Remove nodes that are a parent (and therefore connected)
            disconnected = disconnected.difference(set(node.parent_names))
        disconnected = list(disconnected)
        for name in reversed(disconnected):
            node = self._get_node_by_name(name)
            # Remove nodes that do not have parents
            if len(node.edges) != 0:
                disconnected.remove(node.name)
        names = [n for n in names if not n in disconnected] + [n for n in names if n in disconnected]

        unassigned = names[:]

        # Constructing a valid 'sampling order', which means that the node with no
        # parents will be the first in the sampling order (SO) and so forth.
        sampling_order = []
        while len(sampling_order) < nnodes:
            # Loop over all unassigned nodes
            for i, name in enumerate(unassigned.copy()):
                # Get the parents of the specific nodes
                # parents = set(self.nodes[i].parents)
                unassigned_parents = set(self._get_node_by_name(name).parent_names).difference(sampling_order)
                # If all parents are assigned, the node can be added to the sampling order
                if len(unassigned_parents) == 0:
                    sampling_order.append(name)
                    unassigned.remove(name)

        names = self._node_names
        sampling_order = [names.index(name) for name in sampling_order]

        return sampling_order

    def check_sampling_order(self) -> None:
        """Currently unused"""

        # Get current order
        current_order = self._node_names
        # Get a valid order
        valid_order = self.get_valid_sampling_order()

        # If the sampling order needs to be changed
        if current_order != valid_order:
            # Make sure the correlation matrix is up to date
            # By calculating it
            if len(current_order) > 0:
                self.calculate_correlation_matrix()

        # Change the node list to this order
        self.nodes[:] = [self.nodes[si] for si in valid_order]

        # Change each node's edges to this order
        for node in self.nodes:
            index = [node.parent_index(name) for name in self._node_names if name in node.parent_names]
            node.edges[:] = [node.edges[i] for i in index]

        # If the orders did not match, the conditional correlations need to be changed from the observed correlations

        if current_order != valid_order:
            # First, change the order of the correlation matrix
            names = self._node_names
            order = [current_order.index(node) for node in names]
            self.R = self.R[np.ix_(order, order)]

            # Iterate over all edges, and create conditional correlation from observed correlation, that shouldn't change
            self.create_edge_overview()
            for edge in self.edgelist:
                edge.cond_rank_corr = self.calculate_conditional_correlation(
                    edge.parent,
                    edge.child,
                    observed=self.R[names.index(edge.parent), names.index(edge.child)],
                )

    def change_parent_order(self, source_pos: int, target_pos: int) -> None:
        """Changes the order of the parent nodes, and correspondingly changes the conditional
        rank correlations as well

        Parameters
        ----------
        source_pos : int
            Initial position
        target_pos : int
            Target position
        """
        # Get the current edge overview
        self.create_edge_overview()
        # Add edge on requested position
        self.edgelist.insert(target_pos, self.edgelist.pop(source_pos))
        # Loop through edges and put parents in observed order
        order_changed = False

        for node in self.nodes:
            pnames = node.parent_names
            # Get the order of parents in the user given edge order
            parent_order = [edge.parent for edge in self.edgelist if edge.child == node.name]
            index = [pnames.index(edge) for edge in parent_order]
            # Reorder the edges
            node.edges[:] = [node.edges[i] for i in index]
            # Check if the order changed
            if pnames != node.parent_names:
                order_changed = True
        # Create a new edge overview
        self.create_edge_overview()

        if order_changed:
            # Recalculate the conditional correlations from the observed correlations
            names = self._node_names
            for edge in self.edgelist:
                edge.cond_rank_corr = self.calculate_conditional_correlation(
                    edge.parent,
                    edge.child,
                    observed=self.R[names.index(edge.parent), names.index(edge.child)],
                )

    @property
    def is_dag(self) -> bool:
        """Checks whether the netwerk is a directed acyclic graph"""

        names = self._node_names

        def _check_parents(node, name):
            if name in node.parent_names:
                return False

            for parent_name in node.parent_names:
                parent_node = self.nodes[names.index(parent_name)]
                dag = _check_parents(parent_node, name)
                if not dag:
                    return False

            return True

        for node in self.nodes:
            if not _check_parents(node, node.name):
                return False

        return True

    def calculate_correlation_matrix(self) -> None:
        """Calculates correlation matrix (i.e., non-conditional) from BN and (conditional)
        rank correlations.
        """

        nnodes = len(self.nodes)
        names = self._node_names

        # Initializing the correlation matrix R
        self.R = np.zeros((nnodes, nnodes), dtype=np.float64)
        np.fill_diagonal(self.R, 1.0)

        # Clear the dictionary to save partial correlations that have been calculated
        # and might be used again
        self.partcorrs.clear()

        sampling_order = self.get_valid_sampling_order()
        # Starting the loop for recursively calculating the correlation matrix by
        # the second node (the first sampling node has no parents)
        for i in range(1, nnodes):
            # Get the index in the sampling order
            si = sampling_order[i]

            # Variables for the looping
            cond = []  # Vector storing the conditionalized variables
            T = {}

            parents = [names.index(edge.parent) for edge in self.nodes[si].edges]
            # Contains the previous (same order of SO!) nodes that are not parents
            previous_other = set(sampling_order[:i]).difference(parents)

            # Loop over all parents, and calculate the correlation between the node and its parents
            for edge in self.nodes[si].edges:
                j = names.index(edge.parent)
                rpc = edge.cond_rank_corr

                # Get the rank-correlation to the parent
                T[(si, j)] = ranktopearson(rpc)

                # Get the conditional correlation r1,2;conditions
                s = T[(si, j)]

                # Loop over all conditioning nodes in reverse order, to calculate the unconditional correlation r1,2
                for k in reversed(range(len(cond))):

                    # Recursivelly calculating the correlation between nodes accounting for the conditional/partial correlation
                    r1 = self.calculate_partial_correlation(i=j, j=cond[k], cond=cond[:k])

                    # Based on the conditional/partial correlation, calculating the resulting correlation coefficient
                    # (all the properties of the correlation matrix are guaranteed)
                    shat = s * np.sqrt((1 - T[(si, cond[k])] ** 2) * (1 - (r1) ** 2)) + T[(si, cond[k])] * r1
                    s = shat

                # Saving the correlation coefficients calculated in the upper and lower triangle of the matrix R.
                self.R[si, j] = s
                self.R[j, si] = s

                cond.append(j)

            # Looping over the previous nodes (based on the ordering in SO) which
            # are not parents, stored in previous_other.
            for j in previous_other:
                T[(si, j)] = 0
                s = T[(si, j)]
                for k in reversed(range(len(cond))):
                    if T[(si, cond[k])] != 0 or s != 0:
                        # Recursively calculating the correlation between nodes accounting for the conditional/partial correlation
                        r1 = self.calculate_partial_correlation(j, cond[k], cond[0:k])
                        # Based on the conditional/partial correlation, calculating the resulting correlation coefficient
                        # (all the properties of the correlation matrix are guaranteed)
                        shat = s * np.sqrt((1 - T[(si, cond[k])] ** 2) * (1 - (r1) ** 2)) + T[(si, cond[k])] * r1
                        s = shat

                # Storing the results
                self.R[si, j] = s
                self.R[j, si] = s
                # counter += 1
                cond.append(j)

        self.R = pearsontorank(self.R)

    def calculate_correlation_bounds(self) -> None:
        """Calculates the bounds of the rank correlations in the network. Specified (conditional)
        rank correlation coefficients limit the range of the correlation for other edges. This function
        calculates these limits, by calculating the correlation that would result from imposing a
        conditional -1 and 1 correlation on the edge. This function can be used to display the
        possible range of correlations to a user.
        """

        nnodes = len(self.nodes)
        names = self._node_names

        # Convert to correlation matrix to pearson, for calculating the partial correlations
        self.R = ranktopearson(self.R)

        sampling_order = self.get_valid_sampling_order()
        # Starting the loop for recursively calculating the correlation matrix by
        # the second node
        for i in range(1, nnodes):

            si = sampling_order[i]

            # Variables for the looping
            cond = []  # Vector storing the conditionalized variables
            T = {}  # Dictionary to store temporary conditional correlations

            # Loop over all parents, and calculate the correlation between the node and its parents
            for edge in self.nodes[si].edges:
                pi = names.index(edge.parent)
                rpc = edge.cond_rank_corr

                # Get the (conditional) rank-correlation to the parent
                # First, assume the (conditional) rank-correlation is 1.0, to get the bounds
                s_lower = -1.0
                s_upper = 1.0
                # Get the conditional correlation r1,2;conditions
                # Loop over all conditioning nodes in reverse order, to calculate the unconditional correlation r1,2
                for k in reversed(range(len(cond))):

                    # Recursivelly calculating the correlation between nodes accounting for the conditional/partial correlation
                    r1 = self.calculate_partial_correlation(i=pi, j=cond[k], cond=cond[:k], store=False)

                    # Based on the conditional/partial correlation, calculating the resulting correlation coefficient
                    # (all the properties of the correlation matrix are guaranteed)
                    p1 = np.sqrt((1 - T[(si, cond[k])] ** 2) * (1 - (r1) ** 2))
                    p2 = T[(si, cond[k])] * r1
                    shat_lower = s_lower * p1 + p2
                    s_lower = shat_lower

                    shat_upper = s_upper * p1 + p2
                    s_upper = shat_upper

                if len(cond) == 0:
                    edge.rank_corr_bounds = (-1.0, 1.0)
                    edge.rank_corr = pearsontorank(self.R[si, pi])
                    edge.string = corr_string(r=None, i=pi, j=si, offset=1)
                else:
                    edge.rank_corr_bounds = (
                        pearsontorank(shat_lower),
                        pearsontorank(shat_upper),
                    )
                    edge.rank_corr = pearsontorank(self.R[si, pi])
                    edge.string = corr_string(r=None, i=pi, j=si, cond=sorted(cond), offset=1)

                # Second, set the actual (conditional) rank correlation, to calculate the others
                T[(si, pi)] = ranktopearson(rpc)
                # No need to do any further calculations

                cond.append(pi)

        # Calculate rank correlations again
        self.R = pearsontorank(self.R)

    def calculate_conditional_correlation(self, parent: str, child: str, observed: float) -> float:
        """Calculates conditional correlation give the rank correlation.

        Parameters
        ----------
        parent : str
            Parent node name
        child : str
            Child node name
        observed : float
            Observed rank correlation

        Returns
        -------
        float
            Conditional rank correlation
        """

        names = self._node_names

        # Convert to correlation matrix to pearson, for calculating the partial correlations
        self.R = ranktopearson(self.R)

        si = self._node_names.index(child)
        node = self.nodes[si]

        # Variables for the looping
        cond = []  # Vector storing the conditionalized variables
        T = {}  # Dictionary to store temporary conditional correlations

        # Loop over all parents, and calculate the correlation between the node and its parents
        for edge in node.edges:

            pi = names.index(edge.parent)
            rpc = edge.cond_rank_corr

            # Get the (conditional) rank-correlation to the parent
            # First, assume the (conditional) rank-correlation is 1.0, to get the bounds
            T[(si, pi)] = ranktopearson(rpc)
            # Get the conditional correlation r1,2;conditions
            if edge.parent != parent:
                s = T[(si, pi)]
                # Loop over all conditioning nodes in reverse order, to calculate the unconditional correlation r1,2
                for k in reversed(range(len(cond))):
                    # Recursivelly calculating the correlation between nodes accounting for the conditional/partial correlation
                    r1 = self.calculate_partial_correlation(i=pi, j=cond[k], cond=cond[:k], store=False)
                    # Based on the conditional/partial correlation, calculating the resulting correlation coefficient
                    # (all the properties of the correlation matrix are guaranteed)
                    shat = s * np.sqrt((1 - T[(si, cond[k])] ** 2) * (1 - (r1) ** 2)) + T[(si, cond[k])] * r1
                    s = shat

            # If we reach the requested parent, we reverse the calculation, from a
            # known observed correlation to a conditional correlation
            else:
                s = ranktopearson(observed)
                for k in range(len(cond)):

                    # Recursivelly calculating the correlation between nodes accounting for the conditional/partial correlation
                    r1 = self.calculate_partial_correlation(i=pi, j=cond[k], cond=cond[:k], store=False)
                    # Based on the conditional/partial correlation, calculating the resulting correlation coefficient
                    # (all the properties of the correlation matrix are guaranteed)
                    shat = (s - T[(si, cond[k])] * r1) / np.sqrt((1 - T[(si, cond[k])] ** 2) * (1 - (r1) ** 2))
                    s = shat

                break

            cond.append(pi)

        # Calculate rank correlations again
        self.R = pearsontorank(self.R)

        return pearsontorank(s)

    @property
    def is_invertible(self) -> bool:
        """Whether the correlation matrix is invertible (needed to be a valid correlation matrix)

        Returns
        -------
        bool
            Result
        """
        if self.R.size == 0:
            return True
        return self.R.shape[0] == self.R.shape[1] and np.linalg.matrix_rank(self.R) == self.R.shape[0]

    def to_json(self, path: Path) -> None:
        """Write the BN configuration to JSON

        Parameters
        ----------
        path : Path
            Destination path
        """
        with path.open("w") as f:
            f.write(self.json(indent=4, exclude={"R", "partcorrs", "edgelist"}))

    def calculate_partial_correlation(self, i: int, j: int, cond: List[int], store: bool = True) -> float:
        """
        Calculates the partial correlation with a recursive approach
        and returns both the partial correlations r and a list L containing
        information about the partial correlations during the recursive calculation

        Parameters
        ----------
        i : int
            row index R to calculate the correlation
        j : int
            column index R to calculate the correlation
        cond : list
            conditioning variable(s)
        store : bool
            whether to save the calculated correlation or not (in that case, overwrite)

        Returns
        -------
        r : float
            partial correlation r
        """

        # Defining the number of conditioning variables
        n = len(cond)

        # If the conditioning variable vector is empty, then the value is
        # obtained from the correlation matrix
        if n == 0:
            r = self.R[i, j]
            return r

        # Ordering of the indeces
        if i <= j:
            isort, jsort = i, j
        else:
            isort, jsort = j, i

        # Extracting information on the cell L (from previous calculations)
        comb = (isort, jsort) + tuple(cond)

        # If requested partial correlation is already calculated, return
        if comb in self.partcorrs:
            return self.partcorrs[comb]

        # If not, the correlation still needs to be calculated
        r1 = self.calculate_partial_correlation(i, j, cond[1:n])
        r2 = self.calculate_partial_correlation(i, cond[0], cond[1:n])
        r3 = self.calculate_partial_correlation(j, cond[0], cond[1:n])

        # Calculating partial correlation of [(i,j) | (cond(1),cond(2:n))]
        r = (r1 - r2 * r3) / ((1 - (r2) ** 2) * (1 - (r3) ** 2)) ** (0.5)

        # Limit r to [-1, 1]. Numerical inaccuracies can cause it to be slightly outside this range
        r = max(min(1, r), -1)
        # print(r)
        # assert r >= -1 and r <= 1

        # Saving the results
        if store:
            self.partcorrs[comb] = r

        return r

    def draw_mvn_sample(self, size: int, nodes: list = None) -> np.ndarray:
        """Draw a multivariate normal sample with size for nodes. The sample
        is distributed following the multivariate normal distribution that follows from the BN.

        Parameters
        ----------
        size : int
            Sample size
        nodes : list, optional
            Nodes to include in sample. If None, all are included, by default None

        Returns
        -------
        np.ndarray
            Random sample
        """

        cov = ranktopearson(self.R)
        if nodes is None:
            return np.random.multivariate_normal(mean=np.zeros(len(self.nodes)), cov=cov, size=size)
        else:
            names = self._node_names
            order = [names.index(name) for name in nodes]
            return np.random.multivariate_normal(mean=np.zeros(len(nodes)), cov=cov[np.ix_(order, order)], size=size)
