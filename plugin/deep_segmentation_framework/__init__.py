# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load DeepSegmentationFramework class from file DeepSegmentationFramework.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    #
    from deep_segmentation_framework.deep_segmentation_framework import DeepSegmentationFramework
    return DeepSegmentationFramework(iface)
