import os

# increase limit of pixels (2^30), before importing cv2.
# We are doing it here to make sure it will be done before importing cv2 for the first time
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()


# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load Deepness class from file Deepness.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    from deepness.dialogs.packages_installer import packages_installer_dialog
    packages_installer_dialog.check_required_packages_and_install_if_necessary(iface=iface)

    from deepness.deepness import Deepness
    return Deepness(iface)
