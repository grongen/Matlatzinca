from PyQt5 import Qt, QtCore, QtGui, QtWidgets


def get_width(text):
    width = QtWidgets.QLabel("").fontMetrics().boundingRect(text).width()
    return width


class EnableableWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

    def set_enabled(self, enabled):
        """
        Enable of disable widget elements
        """
        # enable or disable elements in the layout
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i)
            if item.widget():
                item.widget().setEnabled(enabled)


class HLayout(QtWidgets.QHBoxLayout):
    """
    Groups some items in a QHBoxLayout
    """

    def __init__(self, items, stretch=None):
        super(QtWidgets.QHBoxLayout, self).__init__()
        # Add widgets
        if stretch is None:
            stretch = [0] * len(items)
        else:
            if len(items) != len(stretch):
                raise ValueError("Number of given items is not equal to number of given stretchfactors")

        for item, stretchfactor in zip(items, stretch):
            if isinstance(item, str) and item.lower() == "stretch":
                self.addStretch(stretchfactor)
            if isinstance(item, QtWidgets.QLayout):
                self.addLayout(item, stretchfactor)
            elif isinstance(item, QtWidgets.QWidget):
                self.addWidget(item, stretchfactor)
            else:
                TypeError("Item type not understood.")


class VLayout(QtWidgets.QVBoxLayout):
    """
    Groups some items in a QVBoxLayout
    """

    def __init__(self, items, stretch=None):
        super(QtWidgets.QVBoxLayout, self).__init__()
        # Add widgets
        if stretch is None:
            stretch = [0] * len(items)
        else:
            if len(items) != len(stretch):
                raise ValueError("Number of given items is not equal to number of given stretchfactors")

        for item, stretchfactor in zip(items, stretch):
            if isinstance(item, str) and item.lower() == "stretch":
                self.addStretch(stretchfactor)
            if isinstance(item, QtWidgets.QLayout):
                self.addLayout(item, stretchfactor)
            elif isinstance(item, QtWidgets.QWidget):
                self.addWidget(item, stretchfactor)
            else:
                TypeError("Item type not understood.")


class LogoSplitter(QtWidgets.QSplitter):
    """
    Splitter that shows an arrow icon when it is collapsd to one side.
    """

    def __init__(self, topwidget, bottomwidget, toptext="", bottomtext=""):
        """Constructs the widget

        Parameters
        ----------
        topwidget : QWidget
            Widget to display on top
        bottomwidget : QWidget
            Widget to display on bottom
        toptext : str, optional
            Text to display when top is collapsed, by default ''
        bottomtext : str, optional
            Text to display when bottom is collapsed, by default ''
        """
        # Create child class
        QtWidgets.QSplitter.__init__(self, QtCore.Qt.Vertical)

        self.addWidget(topwidget)
        self.addWidget(bottomwidget)

        self.toptext = toptext
        self.bottomtext = bottomtext

        self.splitterMoved.connect(self.on_moved)

        handle_layout = QtWidgets.QVBoxLayout()
        handle_layout.setContentsMargins(0, 0, 0, 0)
        self.setHandleWidth(5)

        self.button = QtWidgets.QToolButton()
        self.button.setStyleSheet("background-color: rgba(255, 255, 255, 0)")

        uplogo = self.style().standardIcon(QtWidgets.QStyle.SP_TitleBarShadeButton)
        downlogo = self.style().standardIcon(QtWidgets.QStyle.SP_TitleBarUnshadeButton)
        self.upicon = QtGui.QIcon(uplogo)
        self.downicon = QtGui.QIcon(downlogo)
        self.noicon = QtGui.QIcon()
        self.icon = self.noicon

        self.button.setIcon(self.icon)
        self.button.clicked.connect(self.handleSplitterButton)
        self.button.setCursor(Qt.QCursor(QtCore.Qt.PointingHandCursor))

        self.label = QtWidgets.QLabel("")

        handle_layout.addLayout(HLayout([self.button, self.label]))
        handle_layout.addItem(
            QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Expanding)
        )

        handle = self.handle(1)
        handle.setLayout(handle_layout)

    def setSizes(self, *args, **kwargs):
        """
        Override the default method to also check the sizes and change label and logo accordingly.
        """
        super().setSizes(*args, **kwargs)
        self.on_moved()

    def handleSplitterButton(self):
        """
        When button is clicked
        """
        # Open and set right icon
        self.setSizes([1, 1])
        self.on_moved()

    def on_moved(self):
        """
        Method to call when splitter is moved. This method
        updates the icon, but it can be extended to update for example a canvs
        """

        sizes = self.sizes()

        # If one is collapsed
        if not sizes[0]:
            icon = self.downicon
            self.label.setText(self.toptext)
            self.setHandleWidth(22 if self.toptext else 5)

        elif not sizes[1]:
            icon = self.upicon
            self.label.setText(self.bottomtext)
            self.setHandleWidth(22 if self.bottomtext else 5)
        # If both are open
        else:
            icon = self.noicon
            self.label.setText("")
            self.setHandleWidth(5)

        if self.icon is not icon:
            self.icon = icon
            self.button.setIcon(self.icon)


class ComboboxInputLine(QtWidgets.QWidget):
    """
    A widget that combines a label and a combo box with several items
    """

    def __init__(self, label, labelwidth=None, items=None, default=None, spacer=True):
        """
        LineEdit class extended with a label in front (description)
        and behind (unit).

        Parameters
        ----------
        label : str
            label for the line edit
        labelwidth : int
            width (points) of label
        items : list
            List with items to add to the combobox
        default : str, optional
            Default item, by default None
        spacer : bool, optional
            Whether or not to add a spacer on the right, by default True
        """

        super(QtWidgets.QWidget, self).__init__()

        self.label = label
        self.labelwidth = labelwidth
        self.items = items
        self.default = default

        self.init_ui(spacer)

        # Add default value
        if self.default is not None:
            if not self.default in self.items:
                raise ValueError("{} not in {}".format(self.default, ", ".join(self.items)))
            else:
                self.combobox.setCurrentText(self.default)

    def init_ui(self, spacer):
        """
        Build ui layout
        """
        self.setLayout(QtWidgets.QHBoxLayout())
        self.layout().setSpacing(10)
        self.layout().setContentsMargins(5, 0, 5, 0)

        # Add label
        self.Label = QtWidgets.QLabel()
        self.Label.setText(self.label)
        if self.labelwidth:
            self.Label.setFixedWidth(self.labelwidth)
        self.layout().addWidget(self.Label)

        # Add line edit
        self.combobox = QtWidgets.QComboBox()
        self.combobox.setMaximumWidth(200)
        self.combobox.addItems(self.items)
        self.layout().addWidget(self.combobox)

        # Add spacer to the right
        if spacer:
            self.layout().addItem(
                QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
            )

    def get_value(self):
        """
        Get value from combobox
        """
        return self.combobox.currentText()

    def set_value(self, value):
        """
        Set value to combobox
        """
        if not isinstance(value, str):
            value = str(value)
        self.combobox.setCurrentText(value)


class ExtendedLineEdit(QtWidgets.QWidget):
    def __init__(self, label, labelwidth=None, browsebutton=False):
        """
        Extended LineEdit class. A browse button can be added, as well as an
        infomessage.

        Parameters
        ----------
        label : str
            label for the line edit
        labelwidth : int
            width (points) of label
        browsebutton : boolean
            Whether to add ad browse button
        info : str
            infomessage to display (not implemented)
        """

        super(QtWidgets.QWidget, self).__init__()
        self.label = label
        self.labelwidth = labelwidth
        self.browsebutton = browsebutton

        self.init_ui()

    def init_ui(self):
        """
        Build ui element
        """

        self.setLayout(QtWidgets.QHBoxLayout())
        self.layout().setSpacing(10)
        self.layout().setContentsMargins(5, 0, 5, 0)

        self.Label = QtWidgets.QLabel()
        self.Label.setText(self.label)
        if self.labelwidth is not None:
            self.Label.setFixedWidth(self.labelwidth)
        self.layout().addWidget(self.Label)

        self.LineEdit = QtWidgets.QLineEdit()
        self.LineEdit.setMinimumWidth(200)
        # self.LineEdit.setReadOnly(True)
        self.layout().addWidget(self.LineEdit)

        if self.browsebutton:
            self.BrowseButton = self.browsebutton
            self.BrowseButton.setFixedWidth(25)
            self.layout().addWidget(self.BrowseButton)

    def get_value(self):
        """
        Get value from line edit
        """
        return self.LineEdit.text()

    def set_value(self, value):
        """
        Set value to line edit
        """
        if not isinstance(value, str):
            value = str(value)
        self.LineEdit.setText(value)


class CheckBoxInput(QtWidgets.QWidget):
    """
    A widget that combines a label and a checkbox
    """

    def __init__(self, label, labelwidth=None, spacer=True, default=False):
        """
        LineEdit class extended with a label in front (description)
        and behind (unit).

        Parameters
        ----------
        label : str
            label for the line edit
        labelwidth : int
            width (points) of label
        items : list
            List with items to add to the combobox
        """
        super(QtWidgets.QWidget, self).__init__()

        self.label = label
        self.labelwidth = labelwidth
        self.init_ui(spacer, default)

    def init_ui(self, spacer, default):
        """
        Build ui layout
        """
        self.setLayout(QtWidgets.QHBoxLayout())
        self.layout().setSpacing(10)
        self.layout().setContentsMargins(5, 0, 5, 0)

        # Add label
        self.Label = QtWidgets.QLabel()
        self.Label.setText(self.label)
        if self.labelwidth:
            self.Label.setFixedWidth(self.labelwidth)
        self.layout().addWidget(self.Label)

        # Add line edit
        self.checkbox = QtWidgets.QCheckBox()
        self.set_value(default)
        self.layout().addWidget(self.checkbox)

        # Add spacer to the right
        if spacer:
            self.layout().addItem(
                QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
            )

    def get_value(self):
        """
        Get value from combobox
        """
        return self.checkbox.isChecked()

    def set_value(self, value):
        """
        Set value to combobox
        """
        return self.checkbox.setChecked(value)


class HLine(QtWidgets.QFrame):
    """
    Adds a horizontal line
    """

    def __init__(self):
        """
        Constructs the line
        """
        super(QtWidgets.QFrame, self).__init__()
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)


class ParameterInputLine(QtWidgets.QWidget):
    """
    A widget that combines a label, input field and unit label
    """

    def __init__(self, label, labelwidth=None, unitlabel=None, validator=None, default=None):
        """
        LineEdit class extended with a label in front (description)
        and behind (unit).

        Parameters
        ----------
        label : str
            label for the line edit
        labelwidth : int
            width (points) of label
        unitlabel : str
            text to add behind line edit
        info : str
            infomessage to display (not implemented)
        """

        super(QtWidgets.QWidget, self).__init__()

        self.label = label
        self.labelwidth = labelwidth
        self.unitlabel = unitlabel
        self.validator = validator
        self.default_value = default
        if default is not None:
            if not isinstance(default, str):
                self.default_value = str(default)

        self.init_ui()

    def init_ui(self):
        """
        Build ui layout
        """
        self.setLayout(QtWidgets.QHBoxLayout())
        self.layout().setSpacing(10)
        self.layout().setContentsMargins(5, 0, 5, 0)

        # Add label
        self.Label = QtWidgets.QLabel()
        self.Label.setText(self.label)
        if self.labelwidth:
            self.Label.setFixedWidth(self.labelwidth)
        self.layout().addWidget(self.Label)

        # Add line edit
        self.LineEdit = QtWidgets.QLineEdit(self.default_value)
        self.LineEdit.setMinimumWidth(40)
        self.LineEdit.setMaximumWidth(150)
        self.LineEdit.setAlignment(QtCore.Qt.AlignRight)
        self.layout().addWidget(self.LineEdit)

        if self.validator is not None:
            self.LineEdit.setValidator(self.validator)

        # Add unit label
        if self.unitlabel is not None:
            self.Label = QtWidgets.QLabel()
            self.Label.setText(self.unitlabel)
            self.layout().addWidget(self.Label)

        # Add spacer to the right
        self.layout().addItem(
            QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        )

    def get_value(self):
        """
        Get value from line edit
        """
        return self.LineEdit.text()

    def set_value(self, value):
        """
        Set value to line edit
        """
        if not isinstance(value, str):
            value = str(value)
        self.LineEdit.setText(value)

    def set_enabled(self, enabled):
        """
        Enable of disable widget elements
        """
        # enable or disable elements in the layout
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i)
            if item.widget():
                item.widget().setEnabled(enabled)
