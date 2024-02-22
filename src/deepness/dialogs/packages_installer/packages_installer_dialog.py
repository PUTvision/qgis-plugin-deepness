"""
This QGIS plugin requires some Python packages to be installed and available.
This tool allows to install them in a local directory, if they are not installed yet.
"""

import importlib
import logging
import os
import subprocess
import sys
import traceback
import urllib
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import List

from qgis.PyQt import QtCore, uic
from qgis.PyQt.QtCore import pyqtSignal
from qgis.PyQt.QtGui import QCloseEvent
from qgis.PyQt.QtWidgets import QDialog, QMessageBox, QTextBrowser

from deepness.common.defines import PLUGIN_NAME

PYTHON_VERSION = sys.version_info
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLUGIN_ROOT_DIR = os.path.realpath(os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..')))
PACKAGES_INSTALL_DIR = os.path.join(PLUGIN_ROOT_DIR, f'python{PYTHON_VERSION.major}.{PYTHON_VERSION.minor}')


FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'packages_installer_dialog.ui'))

_ERROR_COLOR = '#ff0000'


@dataclass
class PackageToInstall:
    name: str
    version: str
    import_name: str  # name while importing package

    def __str__(self):
        return f'{self.name}{self.version}'


REQUIREMENTS_PATH = os.path.join(PLUGIN_ROOT_DIR, 'python_requirements/requirements.txt')

with open(REQUIREMENTS_PATH, 'r') as f:
    raw_txt = f.read()

libraries_versions = {}

for line in raw_txt.split('\n'):
    if line.startswith('#') or not line.strip():
        continue

    line = line.split(';')[0]

    if '==' in line:
        lib, version = line.split('==')
        libraries_versions[lib] = '==' + version
    elif '>=' in line:
        lib, version = line.split('>=')
        libraries_versions[lib] = '>=' + version
    elif '<=' in line:
        lib, version = line.split('<=')
        libraries_versions[lib] = '<=' + version
    else:
        libraries_versions[line] = ''


packages_to_install = [
    PackageToInstall(name='opencv-python-headless', version=libraries_versions['opencv-python-headless'], import_name='cv2'),
]

if sys.platform == "linux" or sys.platform == "linux2":
    packages_to_install += [
        PackageToInstall(name='onnxruntime-gpu', version=libraries_versions['onnxruntime-gpu'], import_name='onnxruntime'),
    ]
    PYTHON_EXECUTABLE_PATH = sys.executable
elif sys.platform == "darwin":  # MacOS
    packages_to_install += [
        PackageToInstall(name='onnxruntime', version=libraries_versions['onnxruntime-gpu'], import_name='onnxruntime'),
    ]
    PYTHON_EXECUTABLE_PATH = str(Path(sys.prefix) / 'bin' / 'python3')  # sys.executable yields QGIS in macOS
elif sys.platform == "win32":
    packages_to_install += [
        PackageToInstall(name='onnxruntime', version=libraries_versions['onnxruntime-gpu'], import_name='onnxruntime'),
    ]
    PYTHON_EXECUTABLE_PATH = 'python'  # sys.executable yields QGis.exe in Windows
else:
    raise Exception("Unsupported operating system!")


class PackagesInstallerDialog(QDialog, FORM_CLASS):
    """
    Dialog witch controls the installation process of packages.
    UI design defined in the `packages_installer_dialog.ui` file.
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

    def move_to_top(self):
        """ Move the window to the top.
        Although if installed from plugin manager, the plugin manager will move itself to the top anyway.
        """
        self.setWindowState((self.windowState() & ~QtCore.Qt.WindowMinimized) | QtCore.Qt.WindowActive)

        if sys.platform == "linux" or sys.platform == "linux2":
            pass
        elif sys.platform == "darwin":  # MacOS
            self.raise_()  # FIXME: this does not really work, the window is still behind the plugin manager
        elif sys.platform == "win32":
            self.activateWindow()
        else:
            raise Exception("Unsupported operating system!")

    def _create_connections(self):
        self.pushButton_close.clicked.connect(self.close)
        self.pushButton_install_packages.clicked.connect(self._run_packages_installation)
        self.signal_log_line.connect(self._log_line)

    def _log_line(self, txt):
        txt = txt \
            .replace('  ', '&nbsp;&nbsp;') \
            .replace('\n', '<br>')
        self.tb.append(txt)

    def log(self, txt):
        self.signal_log_line.emit(txt)

    def _setup_message(self) -> None:
          
        self.log(f'<h2><span style="color: #000080;"><strong>  '
                 f'Plugin {PLUGIN_NAME} - Packages installer </strong></span></h2> \n'
                 f'\n'
                 f'<b>This plugin requires the following Python packages to be installed:</b>')
        
        for package in packages_to_install:
            self.log(f'\t- {package.name}{package.version}')

        self.log('\n\n'
                 f'If this packages are not installed in the global environment '
                 f'(or environment in which QGIS is started) '
                 f'you can install these packages in the local directory (which is included to the Python path).\n\n'
                 f'This Dialog does it for you! (Though you can still install these packages manually instead).\n'
                 f'<b>Please click "Install packages" button below to install them automatically, </b>'
                 f'or "Test and Close" if you installed them manually...\n')

    def _run_packages_installation(self):
        if self.INSTALLATION_IN_PROGRESS:
            self.log(f'Error! Installation already in progress, cannot start again!')
            return
        self.aborted = False
        self.INSTALLATION_IN_PROGRESS = True
        self.thread = Thread(target=self._install_packages)
        self.thread.start()

    def _install_packages(self) -> None:
        self.log('\n\n')
        self.log('=' * 60)
        self.log(f'<h3><b>Attempting to install required packages...</b></h3>')
        os.makedirs(PACKAGES_INSTALL_DIR, exist_ok=True)

        self._install_pip_if_necessary()

        self.log(f'<h3><b>Attempting to install required packages...</b></h3>\n')
        try:
            self._pip_install_packages(packages_to_install)
        except Exception as e:
            msg = (f'\n <span style="color: {_ERROR_COLOR};"><b> '
                   f'Packages installation failed with exception: {e}!\n'
                   f'Please try to install the packages again. </b></span>'
                   f'\nCheck if there is no error related to system packages, '
                   f'which may be required to be installed by your system package manager, e.g. "apt". '
                   f'Copy errors from the stack above and google for possible solutions. '
                   f'Please report these as an issue on the plugin repository tracker!')
            self.log(msg)

        # finally, validate the installation, if there was no error so far...
        self.log('\n\n <b>Installation of required packages finished. Validating installation...</b>')
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
            log_msg += 'And the user confirmed to close the dialog, knowing the risk!'
            event.accept()
        else:
            log_msg += 'The user reconsidered their decision, and will try to install the packages again!'
            event.ignore()
        log_msg += '\n'
        self.log(log_msg)

    def _install_pip_if_necessary(self):
        """
        Install pip if not present.
        It happens e.g. in flatpak applications.

        TODO - investigate whether we can also install pip in local directory
        """

        self.log(f'<h4><b>Making sure pip is installed...</b></h4>')
        if check_pip_installed():
            self.log(f'<em>Pip is installed, skipping installation...</em>\n')
            return

        install_pip_command = [PYTHON_EXECUTABLE_PATH, '-m', 'ensurepip']
        self.log(f'<em>Running command to install pip: \n  $ {" ".join(install_pip_command)} </em>')
        with subprocess.Popen(install_pip_command,
                              stdout=subprocess.PIPE,
                              universal_newlines=True,
                              stderr=subprocess.STDOUT,
                              env={'SETUPTOOLS_USE_DISTUTILS': 'stdlib'}) as process:
            try:
                self._do_process_output_logging(process)
            except InterruptedError as e:
                self.log(str(e))
                return False

        if process.returncode != 0:
            msg = (f'<span style="color: {_ERROR_COLOR};"><b>'
                   f'pip installation failed! Consider installing it manually.'
                   f'<b></span>')
            self.log(msg)
        self.log('\n')

    def _pip_install_packages(self, packages: List[PackageToInstall]) -> None:
        cmd = [PYTHON_EXECUTABLE_PATH, '-m', 'pip', 'install', '-U', f'--target={PACKAGES_INSTALL_DIR}']        
        cmd_string = ' '.join(cmd)
        
        for pck in packages:
            cmd.append(f"{pck}")
            cmd_string += f"{pck}"
        
        self.log(f'<em>Running command: \n  $ {cmd_string} </em>')
        with subprocess.Popen(cmd,
                              stdout=subprocess.PIPE,
                              universal_newlines=True,
                              stderr=subprocess.STDOUT) as process:
            self._do_process_output_logging(process)

        if process.returncode != 0:
            raise RuntimeError('Installation with pip failed')

        msg = (f'\n<b>'
               f'Packages installed correctly!'
               f'<b>\n\n')
        self.log(msg)

    def _do_process_output_logging(self, process: subprocess.Popen) -> None:
        """
        :param process: instance of 'subprocess.Popen'
        """
        for stdout_line in iter(process.stdout.readline, ""):
            if stdout_line.isspace():
                continue
            txt = f'<span style="color: #999999;">{stdout_line.rstrip(os.linesep)}</span>'
            self.log(txt)
            if self.aborted:
                raise InterruptedError('Installation aborted by user')

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
        except Exception:
            msg_base = '<b>Python packages required by the plugin could not be loaded due to the following error:</b>'
            logging.exception(msg_base)
            tb = traceback.format_exc()
            msg1 = (f'<span style="color: {_ERROR_COLOR};">'
                    f'{msg_base} \n '
                    f'{tb}\n\n'
                    f'<b>Please try installing the packages again.<b>'
                    f'</span>')
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
    except Exception:
        logging.exception(f'Python packages required by the plugin could not be loaded due to the following error:')
        return False

    return True


def check_pip_installed() -> bool:
    try:
        subprocess.check_output([PYTHON_EXECUTABLE_PATH, '-m', 'pip', '--version'])
        return True
    except subprocess.CalledProcessError:
        return False


def check_required_packages_and_install_if_necessary(iface):
    os.makedirs(PACKAGES_INSTALL_DIR, exist_ok=True)
    if PACKAGES_INSTALL_DIR not in sys.path:
        sys.path.append(PACKAGES_INSTALL_DIR)  # TODO: check for a less intrusive way to do this

    if are_packages_importable():
        # if packages are importable we are fine, nothing more to do then
        return

    global dialog
    dialog = PackagesInstallerDialog(iface)
    dialog.setWindowModality(QtCore.Qt.WindowModal)
    dialog.show()
    dialog.move_to_top()
