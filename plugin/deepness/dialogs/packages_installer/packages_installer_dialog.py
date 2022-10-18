"""
install plugins in the plugin directory if necessary
"""

import importlib
import logging
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass

from qgis.PyQt.QtWidgets import QMessageBox
from qgis.PyQt import QtGui
from qgis.PyQt.QtGui import QCloseEvent
from qgis.PyQt.QtWidgets import QTextBrowser
from qgis.PyQt import uic
from qgis.PyQt.QtWidgets import QDialogButtonBox
from qgis.PyQt.QtWidgets import QGridLayout
from qgis.PyQt.QtWidgets import QVBoxLayout, QProgressBar, QDialog

from deepness.common.defines import PLUGIN_NAME

PYTHON_VERSION = sys.version_info
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLUGIN_ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
PACKAGES_INSTALL_DIR = os.path.join(PLUGIN_ROOT_DIR, f'python_{PYTHON_VERSION.major}_{PYTHON_VERSION.minor}')
LOG_FILE_PATH = os.path.join(PACKAGES_INSTALL_DIR, 'pip.log')


FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'packages_installer.ui'))


@dataclass
class PackageToInstall:
    name: str
    version: str
    import_name: str  # name while importing package


# NOTE! For the time being requirement are repeated as in `requirements.txt` file
# Consider merging it into some common file in the future
# (if we can use a fixed version of pip packages for all python versions)
packages_to_install = [
    PackageToInstall(name='onnxruntime-gpu', version='1.12.1', import_name='onnxruntime'),
    PackageToInstall(name='opencv-python-headless', version='4.6.0.66', import_name='cv2'),
    PackageToInstall(name='numba', version='4.6.0.66', import_name='numba'),
]


class PackagesInstallerDialog(QDialog, FORM_CLASS):
    """

    UI design defined in the `packages_installer.ui` file.
    """

    def __init__(self, iface, parent=None):
        super(PackagesInstallerDialog, self).__init__(parent)
        self.setupUi(self)
        self.iface = iface
        self.tb = self.textBrowser_log  # type: QTextBrowser
        self._setup_message()
        self._create_connections()

    def _create_connections(self):
        print('hello 88!')
        self.pushButton_close.clicked.connect(self.close)
        self.pushButton_install_packages.clicked.connect(self._install_packages)

    def _setup_message(self):
        required_plugins_str = '\n'.join([f'   - {plugin.name}=={plugin.version}' for plugin in packages_to_install])
        msg1 = f'Plugin {PLUGIN_NAME} - Packages installer \n' \
               f'\n' \
               f'This plugin requires the following Python packages to be installed:\n' \
               f'{required_plugins_str}\n\n' \
               f'If this packages are not installed in the global environment ' \
               f'(or environment in which QGIS is started) ' \
               f'you can install this packages in the local directory and included to the Python path.\n\n' \
               f'This Dialog allows to do it for you!.\n' \
               f'Please click "Install packages" button below...\n'
        self.tb.append(msg1)

    def _install_packages(self):
        msg1 = f'Attempting to install required packages...'
        self.tb.append(msg1)
        for package in packages_to_install:
            msg1 = f'Trying to install "{package.name}"...'
            result = self._install_single_package(package)
            msg1 = f'Package "{package.name}"'
            if result:
                msg1 += ' installed correctly!'
            else:
                msg1 += f' installation failed!' \
                        f'Please try to the install packages again. ' \
                        f'\nCheck if there is no error related to system packages, ' \
                        f'which may be required to be installed by your system package manager, e.g. "apt". ' \
                        f'Copy errors from the stack above into google and look for libraries. ' \
                        f'Please report these as an issue on the plugin repository tracker!'
            self.tb.append(msg1)
            if not result:
                break

        # finally, validate the installation, if there was no error so far...
        self._check_packages_installation_and_log()

    def reject(self) -> None:
        self.close()
        # close_event = QCloseEvent()
        # self.closeEvent(close_event)
        # if close_event.isAccepted():
        #     QDialog.reject()

    def closeEvent(self, event: QCloseEvent):
        if self._check_packages_installation_and_log():
            event.accept()
            return

        res = QMessageBox.question(self.iface.mainWindow(),
                                   f'{PLUGIN_NAME} - skip installation?',
                                   'Are you sure you want to abort the installation of the required python packages? '
                                   'The plugin may not function correctly without them!',
                                   QMessageBox.No, QMessageBox.Yes)
        print(res)
        log_msg = 'User requested to close the dialog, but the packages are not installed correctly!\n'
        if res == QMessageBox.Yes:
            log_msg += 'And the user confirmed to close the dialog, knowing the risc!'
            event.accept()
        else:
            log_msg += 'Fortunately the user reconsidered their decision, and will tr to install the packages again!'
            event.ignore()
        log_msg += '\n'
        self.tb.append(log_msg)

    def _install_single_package(self, package: PackageToInstall) -> bool:
        self.tb.append('\n\n')
        self.tb.append('='*30)
        cmd = ['echo', f"hello\n{package.name}\nTODO\n"]
        popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
        for stdout_line in iter(popen.stdout.readline, ""):
            self.tb.append(stdout_line)
        popen.stdout.close()
        return_code = popen.wait()
        self.tb.append('\n\n')
        if return_code:
            return False
        return True

    def _check_packages_installation_and_log(self) -> bool:
        packages_ok = are_packages_importable()
        self.pushButton_install_packages.setEnabled(not packages_ok)

        if packages_ok:
            msg1 = f'All required packages are importable! You can close this window now!'
            self.tb.append(msg1)
            return True

        try:
            import_packages()
            raise Exception("Unexpected successful import of packages?!? It failed a moment ago, we shouldn't be here!")
        except Exception as e:
            msg_base = 'Python packages required by the plugin could be loaded due to the following error:'
            logging.exception(msg_base)
            tb = traceback.format_exc()
            msg1 = f'{msg_base} \n ' \
                   f'{tb}\n\n' \
                   f'Please try to install the packages again.'
            print(msg1)
            self.tb.append(msg1)

        return False


dialog = None




def import_packages():
    for package in packages_to_install:
        importlib.import_module(package.import_name)


def are_packages_importable() -> bool:
    try:
        import_packages()
    except:
        logging.exception('Python packages required by the plugin could be loaded due to the following error:')
        return False

    return True


def check_required_packages_and_install_if_necessary(iface):
    if are_packages_importable():
        # if packages are importable we are fine, nothing more to do then
        return

    global dialog
    dialog = PackagesInstallerDialog(iface)
    dialog.show()
