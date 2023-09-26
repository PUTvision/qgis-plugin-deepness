"""
This is an integration test of multiple components.
"""

import sys
from test.test_utils import (SignalCollector, create_rlayer_from_file, create_vlayer_from_file,
                             get_dummy_fotomap_area_path, get_dummy_fotomap_small_path,
                             get_dummy_segmentation_model_path, init_qgis)
from unittest.mock import MagicMock

import pytest
from qgis.core import (QgsApplication, QgsCoordinateReferenceSystem, QgsProject, QgsRasterLayer, QgsRectangle,
                       QgsVectorLayer)
from qgis.PyQt.QtWidgets import QApplication

from deepness.common.channels_mapping import ChannelsMapping
from deepness.common.config_entry_key import ConfigEntryKey
from deepness.common.processing_parameters.map_processing_parameters import (MapProcessingParameters, ModelOutputFormat,
                                                                             ProcessedAreaType)
from deepness.common.processing_parameters.segmentation_parameters import SegmentationParameters
from deepness.common.processing_parameters.training_data_export_parameters import TrainingDataExportParameters
from deepness.deepness_dockwidget import DeepnessDockWidget
from deepness.processing.models.model_types import ModelType
from deepness.processing.models.segmentor import Segmentor

RASTER_FILE_PATH = get_dummy_fotomap_small_path()
VLAYER_MASK_FILE_PATH = get_dummy_fotomap_area_path()
MODEL_FILE_PATH = get_dummy_segmentation_model_path()


def test_run_inference():
    qgs = init_qgis()

    # with 2 example filed from config to validate the logic flow

    ConfigEntryKey.PROCESSED_AREA_TYPE.set(ProcessedAreaType.VISIBLE_PART.value)
    ConfigEntryKey.MODEL_FILE_PATH.set(MODEL_FILE_PATH)
    ConfigEntryKey.PREPROCESSING_TILES_OVERLAP.set(44)

    dockwidget = DeepnessDockWidget(iface=MagicMock())
    dockwidget._get_input_layer_id = MagicMock(return_value=1)  # fake input layer id, just to test

    # set to different values to check if will be saved while running ui
    ConfigEntryKey.PROCESSED_AREA_TYPE.set(ProcessedAreaType.ENTIRE_LAYER.value)
    ConfigEntryKey.PREPROCESSING_TILES_OVERLAP.set(55)

    signal_collector = SignalCollector(dockwidget.run_model_inference_signal)
    dockwidget.pushButton_runInference.click()

    assert signal_collector.was_called
    params = signal_collector.get_first_arg()
    assert isinstance(params, MapProcessingParameters)
    assert ConfigEntryKey.PROCESSED_AREA_TYPE.get() == ProcessedAreaType.VISIBLE_PART.value
    assert ConfigEntryKey.PREPROCESSING_TILES_OVERLAP.get() == 44


def test_run_data_export():
    qgs = init_qgis()

    dockwidget = DeepnessDockWidget(iface=MagicMock())
    dockwidget._get_input_layer_id = MagicMock(return_value=1)  # fake input layer id, just to test

    signal_collector = SignalCollector(dockwidget.run_training_data_export_signal)
    dockwidget.pushButton_runTrainingDataExport.click()

    assert signal_collector.was_called
    params = signal_collector.get_first_arg()
    assert isinstance(params, TrainingDataExportParameters)


def test_get_inference_parameters():
    qgs = init_qgis()

    rlayer = create_rlayer_from_file(RASTER_FILE_PATH)
    vlayer_mask = create_vlayer_from_file(VLAYER_MASK_FILE_PATH)
    ConfigEntryKey.MODEL_TYPE.set(ModelType.SEGMENTATION.value)
    ConfigEntryKey.MODEL_FILE_PATH.set(MODEL_FILE_PATH)
    ConfigEntryKey.PREPROCESSING_RESOLUTION.set(7)
    ConfigEntryKey.PROCESSED_AREA_TYPE.set(ProcessedAreaType.VISIBLE_PART.value)
    ConfigEntryKey.PREPROCESSING_TILES_OVERLAP.set(44)
    ConfigEntryKey.MODEL_OUTPUT_FORMAT.set(ModelOutputFormat.ONLY_SINGLE_CLASS_AS_LAYER.value)
    ConfigEntryKey.MODEL_OUTPUT_FORMAT_CLASS_NUMBER.set(1)

    dockwidget = DeepnessDockWidget(iface=MagicMock())
    dockwidget._get_input_layer_id = MagicMock(return_value=1)  # fake input layer id, just to test

    params = dockwidget.get_inference_parameters()
    assert isinstance(params, SegmentationParameters)

    assert isinstance(params.model, Segmentor)
    assert params.resolution_cm_per_px == 7
    assert params.processed_area_type == ProcessedAreaType.VISIBLE_PART
    assert params.tile_size_px == 512  # should be read from model input
    # assert params.input_layer_id == rlayer.id()
    assert params.processing_overlap.get_overlap_px(params.tile_size_px) == int(0.44*params.tile_size_px)
    assert params.input_channels_mapping.get_number_of_model_inputs() == 3
    assert params.input_channels_mapping.get_number_of_image_channels() == 4
    assert params.input_channels_mapping.get_image_channel_index_for_model_input(2) == 2
    assert params.model_output_format == ModelOutputFormat.ONLY_SINGLE_CLASS_AS_LAYER
    assert params.model_output_format__single_class_number == 1


if __name__ == '__main__':
    test_run_inference()
    test_run_data_export()
    test_get_inference_parameters()
    print('Done')
