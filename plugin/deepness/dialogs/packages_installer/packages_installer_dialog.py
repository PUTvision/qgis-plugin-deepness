import os
import sys

from qgis.PyQt.QtWidgets import QDialogButtonBox
from qgis.PyQt.QtWidgets import QGridLayout
from qgis.PyQt.QtWidgets import QVBoxLayout, QProgressBar, QDialog

"""
install plugins in the plugin directory
"""

PYTHON_VERSION = sys.version_info
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLUGIN_ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
PACKAGES_INSTALL_DIR = os.path.join(PLUGIN_ROOT_DIR, f'python_{PYTHON_VERSION.major}_{PYTHON_VERSION.minor}')
LOG_FILE_PATH = os.path.join(PACKAGES_INSTALL_DIR, 'pip.log')


class MyDialog(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.setLayout(QGridLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.buttonbox = QDialogButtonBox(QDialogButtonBox.Ok)
        self.buttonbox.accepted.connect(self.run)
        self.layout().addWidget(self.buttonbox, 0, 0, 2, 1)

    def run(self):
        print('hello')

# function to import packages to another file?


def check_required_packages_and_install_if_necessary():
    print('hello1')
    dialog = MyDialog()
    dialog.show()
