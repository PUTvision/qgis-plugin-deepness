# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load Deepness class from file Deepness.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    #
    from deepness.deepness import Deepness
    return Deepness(iface)
