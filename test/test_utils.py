import os

from qgis.PyQt.QtWidgets import QWidget
from qgis.core import QgsVectorLayer, QgsProject
from qgis.core import QgsCoordinateReferenceSystem, QgsRectangle, QgsApplication
from qgis.core import QgsRasterLayer

from deep_segmentation_framework.common.channels_mapping import ChannelsMapping, ImageChannelStandaloneBand, \
    ImageChannelCompositeByte

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, 'data'))


def get_dummy_segmentation_model_path():
    """
    Get path of a dummy onnx model. See details in README in model directory.
    Model used for unit tests processing purposes
    """
    return os.path.join(TEST_DATA_DIR, 'dummy_model', 'dummy_segmentation_model.onnx')


def get_dummy_regression_model_path():
    """
    Get path of a dummy onnx model. See details in README in model directory.
    Model used for unit tests processing purposes
    """
    return os.path.join(TEST_DATA_DIR, 'dummy_model', 'dummy_regression_model.onnx')


def get_dummy_fotomap_small_path():
    """
    Get path of dummy fotomap tif file, which can be used
    for testing with conjunction with dummy_mode (see get_dummy_model_path)
    """
    return os.path.join(TEST_DATA_DIR, 'dummy_fotomap_small.tif')


def get_dummy_fotomap_area_path():
    """
    Get path of the file with processing area polygon, for dummy_fotomap (see get_dummy_fotomap_small_path)
    """
    return os.path.join(TEST_DATA_DIR, 'dummy_fotomap_area.gpkg')


def create_rlayer_from_file(file_path):
    """
    Create raster layer from tif file and add it to current QgsProject
    """
    rlayer = QgsRasterLayer(file_path, 'fotomap')
    if rlayer.width() == 0:
        raise Exception("0 width - rlayer not loaded properly. Probably invalid file path?")
    rlayer.setCrs(QgsCoordinateReferenceSystem("EPSG:32633"))
    QgsProject.instance().addMapLayer(rlayer)
    return rlayer


def create_vlayer_from_file(file_path):
    """
    Create vector layer from geometry file and add it to current QgsProject
    """
    vlayer = QgsVectorLayer(file_path)
    if not vlayer.isValid():
        raise Exception("Invalid vlayer! Probably invalid file path?")
    QgsProject.instance().addMapLayer(vlayer)
    return vlayer


def create_default_input_channels_mapping_for_rgba_bands():
    # as in 'set_rlayer' function in 'input_channels_mapping_widget'

    channels_mapping = ChannelsMapping()
    channels_mapping.set_image_channels(
        [
            ImageChannelStandaloneBand(band_number=1, name='red'),
            ImageChannelStandaloneBand(band_number=2, name='green'),
            ImageChannelStandaloneBand(band_number=3, name='blue'),
            ImageChannelStandaloneBand(band_number=4, name='alpha'),
        ]
    )
    channels_mapping.set_number_of_model_inputs_same_as_image_channels()
    return channels_mapping


def create_default_input_channels_mapping_for_rgb_bands():
    # as in 'set_rlayer' function in 'input_channels_mapping_widget'

    channels_mapping = ChannelsMapping()
    channels_mapping.set_image_channels(
        [
            ImageChannelStandaloneBand(band_number=1, name='red'),
            ImageChannelStandaloneBand(band_number=2, name='green'),
            ImageChannelStandaloneBand(band_number=3, name='blue'),
        ]
    )
    channels_mapping.set_number_of_model_inputs_same_as_image_channels()
    return channels_mapping


def create_default_input_channels_mapping_for_google_satellite_bands():
    # as in 'set_rlayer' function in 'input_channels_mapping_widget'

    channels_mapping = ChannelsMapping()
    channels_mapping.set_number_of_model_inputs(3)
    channels_mapping.set_image_channels(
        [
            ImageChannelCompositeByte(byte_number=2, name='red'),
            ImageChannelCompositeByte(byte_number=1, name='green'),
            ImageChannelCompositeByte(byte_number=0, name='blue'),
            ImageChannelCompositeByte(byte_number=3, name='alpha'),
        ]
    )
    return channels_mapping


class SignalCollector(QWidget):
    """
    Allows to intercept a signal and collect its data during unit testing
    """

    def __init__(self, signal_to_collect):
        super().__init__()
        self.was_called = False
        self.signal_args = None
        self.signal_kwargs = None
        signal_to_collect.connect(self.any_slot)

    def any_slot(self, *args, **kwargs):
        self.signal_kwargs = kwargs
        self.signal_args = args
        self.was_called = True

    def get_first_arg(self):
        assert self.was_called
        if self.signal_args:
            return self.signal_args[0]
        if self.signal_kwargs:
            return list(self.signal_kwargs.values())[0]
        raise Exception("No argument were provided for the signal!")


def init_qgis():
    qgs = QgsApplication([b''], False)
    qgs.setPrefixPath('/usr/bin/qgis', True)
    qgs.initQgis()
    return qgs
