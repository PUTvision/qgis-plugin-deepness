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
from dataclasses import dataclass
from threading import Thread

from qgis.PyQt.QtCore import pyqtSignal
from qgis.PyQt.QtWidgets import QMessageBox
from qgis.PyQt.QtGui import QCloseEvent
from qgis.PyQt.QtWidgets import QTextBrowser
from qgis.PyQt import uic, QtCore
from qgis.PyQt.QtWidgets import QDialog

from deepness.common.defines import PLUGIN_NAME

PYTHON_VERSION = sys.version_info
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLUGIN_ROOT_DIR = os.path.realpath(os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..')))
PACKAGES_INSTALL_DIR = os.path.join(PLUGIN_ROOT_DIR, f'python{PYTHON_VERSION.major}.{PYTHON_VERSION.minor}')


FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'packages_installer_dialog.ui'))


@dataclass
class PackageToInstall:
    name: str
    version: str
    import_name: str  # name while importing package


# NOTE! For the time being requirement are repeated as in `requirements.txt` file
# Consider merging it into some common file in the future
# (if we can use a fixed version of pip packages for all python versions)
packages_to_install = [
    PackageToInstall(name='opencv-python-headless', version='4.6.0.66', import_name='cv2'),
]

if sys.platform == "linux" or sys.platform == "linux2":
    packages_to_install += [
        PackageToInstall(name='onnxruntime-gpu', version='1.12.1', import_name='onnxruntime'),
    ]
elif sys.platform == "darwin":  # MacOS
    packages_to_install += [
        PackageToInstall(name='onnxruntime', version='1.12.1', import_name='onnxruntime'),
    ]
elif sys.platform == "win32":
    packages_to_install += [
        PackageToInstall(name='onnxruntime', version='1.12.1', import_name='onnxruntime'),
    ]
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
            self.raise_()
        elif sys.platform == "win32":
            self.activateWindow()
        else:
            raise Exception("Unsupported operating system!")

    def _create_connections(self):
        self.pushButton_close.clicked.connect(self.close)
        self.pushButton_install_packages.clicked.connect(self._run_packages_installation)
        self.signal_log_line.connect(self._log_line)

    def _log_line(self, txt):
        txt = txt\
            .replace('  ', '&nbsp;&nbsp;')\
            .replace('\n', '<br>')
        self.tb.append(txt)

    def log(self, txt):
        self.signal_log_line.emit(txt)

    def _setup_message(self):
        required_plugins_str = '\n'.join([f'   - {plugin.name}=={plugin.version}' for plugin in packages_to_install])
        msg1 = f'<h2><span style="color: #000080;"><strong>  ' \
               f'Plugin {PLUGIN_NAME} - Packages installer </strong></span></h2> \n' \
               f'\n' \
               f'<b>This plugin requires the following Python packages to be installed:</b>\n' \
               f'{required_plugins_str}\n\n' \
               f'If this packages are not installed in the global environment ' \
               f'(or environment in which QGIS is started) ' \
               f'you can install these packages in the local directory (which is included to the Python path).\n\n' \
               f'This Dialog allows to do it for you! (Though you can still install these packages manually instead).\n' \
               f'<b>Please click "Install packages" button below to install them automatically, </b>' \
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
        self.log(f'<h3><b>Attempting to install required packages...</b></h3>\n')
        os.makedirs(PACKAGES_INSTALL_DIR, exist_ok=True)

        self._install_pip_if_necessary()

        self.log(f'<h3><b>Attempting to install required packages...</b></h3>\n')

        for package in packages_to_install:
            if self.aborted:
                break
            self.log(f'<b> &rarr; Trying to install "{package.name}"... </b>')
            result = self._install_single_package(package)
            if not result:
                msg = f'\n <span style="color: #ff0000;"><b> ' \
                      f'Package "{package.name} installation failed!\n' \
                      f'Please try to the install packages again. </b></span>' \
                      f'\nCheck if there is no error related to system packages, ' \
                      f'which may be required to be installed by your system package manager, e.g. "apt". ' \
                      f'Copy errors from the stack above into google and look for libraries. ' \
                      f'Please report these as an issue on the plugin repository tracker!'
                self.log(msg)
                break

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
            log_msg += 'And the user confirmed to close the dialog, knowing the risc!'
            event.accept()
        else:
            log_msg += 'Fortunately the user reconsidered their decision, and will tr to install the packages again!'
            event.ignore()
        log_msg += '\n'
        self.log(log_msg)

    def _install_pip_if_necessary(self):
        """
        Install pip if not present.
        It happens e.g. in flatpack applications.

        TODO - investigate whether we can also install pip in local directory
        """

        self.log(f'<h4><b>Making sure pip is installed...</b></h4>')
        p = subprocess.Popen(['python', '-m', 'pip', '--version'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (output, err) = p.communicate()
        if p.returncode == 0:
            self.log(f'<b> pip is installed and available! <b>\n')
            return
        else:
            self.log(f'<b> pip is not available! <b>\n')

        # install pip. Environmental variable added due to a bug - see https://github.com/pypa/setuptools/issues/2941
        install_pip_command = "export SETUPTOOLS_USE_DISTUTILS=stdlib && python -m ensurepip"
        self.log(f'<em>Running command to install pip: \n  $ {install_pip_command} </em>')
        popen = subprocess.Popen(install_pip_command,
                                 stdout=subprocess.PIPE,
                                 universal_newlines=True,
                                 stderr=subprocess.STDOUT,
                                 shell=True)

        return_code = self._wait_for_process_to_finish_with_logging(popen)
        if self.aborted:
            return False

        if return_code != 0:
            msg = f'<span style="color: #ff0000;"><b>' \
                  f'pip installation failed!' \
                  f'<b></span>\n'
            self.log(msg)

    def _install_single_package(self, package: PackageToInstall) -> bool:
        try:
            import_package(package)
            self.log(f'  No need to install, "{package.name}" already installed\n\n')
            return True
        except:
            pass  # we are going to install the package below

        cmd = ['python', '-m', 'pip', f"install", f"--target={PACKAGES_INSTALL_DIR}", f"{package.name}"]
        msg = ' '.join(cmd)
        self.log(f'<em>Running command: \n  $ {msg} </em>')
        popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True, stderr=subprocess.STDOUT)

        self.log('*'*30)
        return_code = self._wait_for_process_to_finish_with_logging(popen)
        self.log('*'*30)
        if self.aborted:
            return False

        if return_code != 0:
            return False

        msg = f'\n<b>' \
              f'Package "{package.name}" installed correctly!' \
              f'<b>\n\n'
        self.log(msg)

        return True

    def _wait_for_process_to_finish_with_logging(self, popen) -> int:
        """
        :param popen: instance of 'subprocess.Popen'
        :return: process return code
        """
        for stdout_line in iter(popen.stdout.readline, ""):
            if stdout_line.isspace():
                continue
            stdout_line = '    ' + stdout_line.strip('\n')
            txt = f'<span style="color: #999999;"> {stdout_line} </span>'
            self.log(txt)
            if self.aborted:
                self.log('Error! Installation aborted!')
                return -1

        popen.stdout.close()
        return_code = popen.wait()
        return return_code

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
            msg_base = '<b>Python packages required by the plugin could be loaded due to the following error:</b>'
            logging.exception(msg_base)
            tb = traceback.format_exc()
            msg1 = f'<span style="color: #ff0000;">' \
                   f'{msg_base} \n ' \
                   f'{tb}\n\n' \
                   f'<b>Please try to install the packages again.<b>' \
                   f'</span>'
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
    os.makedirs(PACKAGES_INSTALL_DIR, exist_ok=True)
    if PACKAGES_INSTALL_DIR not in sys.path:
        sys.path.append(PACKAGES_INSTALL_DIR)

    if are_packages_importable():
        # if packages are importable we are fine, nothing more to do then
        return

    global dialog
    dialog = PackagesInstallerDialog(iface)
    dialog.setWindowModality(QtCore.Qt.WindowModal)
    dialog.show()
    dialog.move_to_top()
