"""
This QGIS plugin requires some Python packages to be installed and available.
This tool allows to install them in a local directory, if they are not installed yet.
"""

import importlib
import logging
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from threading import Thread

from qgis.PyQt.QtCore import pyqtSignal
from qgis.PyQt.QtWidgets import QMessageBox
from qgis.PyQt.QtGui import QCloseEvent
from qgis.PyQt.QtWidgets import QTextBrowser
from qgis.PyQt import uic
from qgis.PyQt.QtWidgets import QVBoxLayout, QProgressBar, QDialog

from deepness.common.defines import PLUGIN_NAME

PYTHON_VERSION = sys.version_info
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLUGIN_ROOT_DIR = os.path.realpath(os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..')))
PACKAGES_INSTALL_DIR = os.path.join(PLUGIN_ROOT_DIR, f'python{PYTHON_VERSION.major}.{PYTHON_VERSION.minor}')


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
]


class PackagesInstallerDialog(QDialog, FORM_CLASS):
    """
    Dialog witch controls the installation process of packages.
    UI design defined in the `packages_installer.ui` file.
    """

    signal_log_line = pyqtSignal(str)  # we need to use signal because we cannot edit GUI from another thread

    INSTALLATION_IN_PROGRESS = False  # to make sure we will not start the installation twice

    def __init__(self, iface, parent=None):
        super(PackagesInstallerDialog, self).__init__(parent)
        self.setupUi(self)
        self.iface = iface
        self.tb = self.textBrowser_log  # type: QTextBrowser
        self._create_connections()
        self._setup_message()
        self.aborted = False
        self.thread = None

    def _create_connections(self):
        self.pushButton_close.clicked.connect(self.close)
        self.pushButton_install_packages.clicked.connect(self._run_packages_installation)
        self.signal_log_line.connect(self._log_line)

    def _log_line(self, txt):
        self.tb.append(txt)

    def log(self, txt):
        self.signal_log_line.emit(txt)

    def _setup_message(self):
        required_plugins_str = '\n'.join([f'   - {plugin.name}=={plugin.version}' for plugin in packages_to_install])
        msg1 = f'Plugin {PLUGIN_NAME} - Packages installer \n' \
               f'\n' \
               f'This plugin requires the following Python packages to be installed:\n' \
               f'{required_plugins_str}\n\n' \
               f'If this packages are not installed in the global environment ' \
               f'(or environment in which QGIS is started) ' \
               f'you can install these packages in the local directory (which is included to the Python path).\n\n' \
               f'This Dialog allows to do it for you! (Though you can still install these packages manually instead).\n' \
               f'Please click "Install packages" button below to install them automatically, ' \
               f'or "Test and Close" if you installed them manually...\n'
        self.log(msg1)

    def _run_packages_installation(self):
        if self.INSTALLATION_IN_PROGRESS:
            self.log(f'Error! Installation already in progress, cannot start again!')
            return
        self.aborted = False
        self.INSTALLATION_IN_PROGRESS = True
        self.thread = Thread(target=self._install_packages)
        self.thread.start()

    def _install_packages(self):
        self.log('\n\n')
        self.log('='*60)
        self.log(f'Attempting to install required packages...\n')
        os.makedirs(PACKAGES_INSTALL_DIR, exist_ok=True)

        for package in packages_to_install:
            if self.aborted:
                break
            self.log(f'### Trying to install "{package.name}"...')
            result = self._install_single_package(package)
            if not result:
                msg = f'Package "{package.name} installation failed!' \
                      f'Please try to the install packages again. ' \
                      f'\nCheck if there is no error related to system packages, ' \
                      f'which may be required to be installed by your system package manager, e.g. "apt". ' \
                      f'Copy errors from the stack above into google and look for libraries. ' \
                      f'Please report these as an issue on the plugin repository tracker!'
                self.log(msg)
                break

        # finally, validate the installation, if there was no error so far...
        self.log('\n\n Installation of required packages finished. Validating installation...')
        self._check_packages_installation_and_log()
        self.INSTALLATION_IN_PROGRESS = False

    def reject(self) -> None:
        self.close()

    def closeEvent(self, event: QCloseEvent):
        self.aborted = True
        if self._check_packages_installation_and_log():
            event.accept()
            return

        res = QMessageBox.question(self.iface.mainWindow(),
                                   f'{PLUGIN_NAME} - skip installation?',
                                   'Are you sure you want to abort the installation of the required python packages? '
                                   'The plugin may not function correctly without them!',
                                   QMessageBox.No, QMessageBox.Yes)
        log_msg = 'User requested to close the dialog, but the packages are not installed correctly!\n'
        if res == QMessageBox.Yes:
            log_msg += 'And the user confirmed to close the dialog, knowing the risc!'
            event.accept()
        else:
            log_msg += 'Fortunately the user reconsidered their decision, and will tr to install the packages again!'
            event.ignore()
        log_msg += '\n'
        self.log(log_msg)

    def _install_single_package(self, package: PackageToInstall) -> bool:
        try:
            import_package(package)
            self.log(f'  No need to install, "{package.name}" already installed\n\n')
            return True
        except:
            pass  # we are going to install the package below

        cmd = ['pip', f"install", f"--target={PACKAGES_INSTALL_DIR}", f"{package.name}"]
        msg = ' '.join(cmd)
        self.log(f'Running command: \n  $ "{msg}"')
        # cmd = ['echo', f"hello\n{package.name}\nTODO\n"]
        popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
        for stdout_line in iter(popen.stdout.readline, ""):
            self.log(stdout_line)
            time.sleep(0.01)
            if self.aborted:
                self.log('Error! Installation aborted!')
                return False

        popen.stdout.close()
        return_code = popen.wait()
        if return_code:
            return False

        msg = f'Package "{package.name}" installed correctly!\n\n'
        self.log(msg)

        return True

    def _check_packages_installation_and_log(self) -> bool:
        packages_ok = are_packages_importable()
        self.pushButton_install_packages.setEnabled(not packages_ok)

        if packages_ok:
            msg1 = f'All required packages are importable! You can close this window now!'
            self.log(msg1)
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
            self.log(msg1)

        return False


dialog = None


def import_package(package: PackageToInstall):
    importlib.import_module(package.import_name)


def import_packages():
    for package in packages_to_install:
        import_package(package)


def are_packages_importable() -> bool:
    try:
        import_packages()
    except:
        logging.exception('Python packages required by the plugin could be loaded due to the following error:')
        return False

    return True


def check_required_packages_and_install_if_necessary(iface):
    print(f'{PACKAGES_INSTALL_DIR = }')
    if PACKAGES_INSTALL_DIR not in sys.path:
        os.makedirs(PACKAGES_INSTALL_DIR, exist_ok=True)
        sys.path.append(PACKAGES_INSTALL_DIR)

    if are_packages_importable():
        # if packages are importable we are fine, nothing more to do then
        return

    global dialog
    dialog = PackagesInstallerDialog(iface)
    dialog.show()
