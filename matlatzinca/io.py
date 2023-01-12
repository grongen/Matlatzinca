import csv
from itertools import product

import numpy as np
from PyQt5 import QtCore


def selection_to_text(selection, newline="\n", delimiter="\t", newline_replace=None):
    """
    Get elements from model selection and convert to text

    Parameters
    ----------
    selection : Qt selection
        Selection from a table
    newline : str, optional
        character for newline, by default \n
    delimiter : str, optional
        character for delimiter, by default \t

    Returns
    -------
    str
        Text that can be exported or pasted
    """
    rows = [index.row() for index in selection]
    cols = [index.column() for index in selection]
    minrow, maxrow = min(rows), max(rows)
    mincol, maxcol = min(cols), max(cols)

    # Create empty Nrow*Ncol table to fill the selected values
    rowcount = maxrow - minrow + 1
    colcount = maxcol - mincol + 1
    table = [[""] * colcount for _ in range(rowcount)]

    # Fill all the selected values
    for index in selection:
        row = index.row() - minrow
        column = index.column() - mincol
        item = index.data()
        table[row][column] = item if isinstance(item, str) else ""

    if newline_replace is None:
        newline_replace = newline

    return newline.join(
        [delimiter.join([item.replace(newline, newline_replace) for item in row]) for row in table]
    )


def get_table_text(model, newline="\n", delimiter="\t", newline_replace=None):
    """
    Get all elements from table model, including the header

    Parameters
    ----------
    selection : Qt selection
        Selection from a table
    newline : str, optional
        character for newline, by default \n
    delimiter : str, optional
        character for delimiter, by default \t

    Returns
    -------
    str
        Text that can be exported or pasted
    """
    # Get header
    header = [
        model.headerData(i, QtCore.Qt.Horizontal, QtCore.Qt.DisplayRole)
        for i in range(model.columnCount(None))
    ]
    if header[0] == "ID":
        header[0] = "Id"
    # Replace newline characters in the header
    if newline_replace is not None:
        header = [item.replace(newline, newline_replace) for item in header]
    # Get data text
    selection = [
        model.createIndex(i, j)
        for i, j in product(range(model.rowCount(None)), range(model.columnCount(None)))
    ]
    return delimiter.join(header) + newline + selection_to_text(selection, newline, delimiter) + newline
