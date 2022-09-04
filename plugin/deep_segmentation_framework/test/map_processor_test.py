import sys
from unittest.mock import MagicMock

import pytest
from qgis.PyQt.QtWidgets import QApplication
from qgis.core import QgsVectorLayer, QgsProject
from qgis.core import QgsCoordinateReferenceSystem, QgsRectangle, QgsApplication
from qgis.core import QgsRasterLayer

from deep_segmentation_framework.common.channels_mapping import ChannelsMapping
from deep_segmentation_framework.common.inference_parameters import ProcessedAreaType, InferenceParameters
from deep_segmentation_framework.deep_segmentation_framework_dockwidget import DeepSegmentationFrameworkDockWidget
from deep_segmentation_framework.processing.map_processor import MapProcessor
from deep_segmentation_framework.processing.model_wrapper import ModelWrapper
from deep_segmentation_framework.test.test_utils import init_qgis, create_rlayer_from_file, \
    create_vlayer_from_file, get_dummy_fotomap_area_path, get_dummy_fotomap_small_path, get_dummy_model_path, \
    create_default_input_channels_mapping_for_rgba_bands

RASTER_FILE_PATH = get_dummy_fotomap_small_path()

VLAYER_MASK_FILE_PATH = get_dummy_fotomap_area_path()

MODEL_FILE_PATH = get_dummy_model_path()

INPUT_CHANNELS_MAPPING = create_default_input_channels_mapping_for_rgba_bands()

PROCESSED_EXTENT_1 = QgsRectangle(  # big part of the fotomap
        638840.370, 5802593.197,
        638857.695, 5802601.792,)


def dummy_model_processing__entire_file():
    qgs = init_qgis()

    rlayer = create_rlayer_from_file(RASTER_FILE_PATH)
    model_wrapper = ModelWrapper(MODEL_FILE_PATH)

    inference_parameters = InferenceParameters(
        resolution_cm_per_px=3,
        tile_size_px=model_wrapper.get_input_size_in_pixels()[0],  # same x and y dimensions, so take x
        processed_area_type=ProcessedAreaType.ENTIRE_LAYER,
        mask_layer_id=None,
        input_layer_id=rlayer.id(),
        input_channels_mapping=INPUT_CHANNELS_MAPPING,
        postprocessing_dilate_erode_size=5,
        processing_overlap_percentage=20,
        model=model_wrapper,
    )

    map_processor = MapProcessor(
        rlayer=rlayer,
        vlayer_mask=None,
        map_canvas=MagicMock(),
        inference_parameters=inference_parameters,
    )

    map_processor.run()
    result_img = map_processor.get_result_img()

    assert result_img.shape == (561, 829)
    # TODO - add detailed check for pixel values once we have output channels mapping with thresholding


def generic_processing_test__specified_extent_from_vlayer():
    qgs = init_qgis()

    rlayer = create_rlayer_from_file(RASTER_FILE_PATH)
    vlayer_mask = create_vlayer_from_file(VLAYER_MASK_FILE_PATH)
    vlayer_mask.setCrs(rlayer.crs())
    model_wrapper = MagicMock()
    model_wrapper.process = lambda x: x[:, :, 0]
    model_wrapper.get_number_of_channels = lambda: 3

    inference_parameters = InferenceParameters(
        resolution_cm_per_px=3,
        tile_size_px=512,
        processed_area_type=ProcessedAreaType.FROM_POLYGONS,
        mask_layer_id=vlayer_mask.id(),
        input_layer_id=rlayer.id(),
        input_channels_mapping=INPUT_CHANNELS_MAPPING,
        postprocessing_dilate_erode_size=5,
        processing_overlap_percentage=20,
        model=model_wrapper,
    )
    map_processor = MapProcessor(
        rlayer=rlayer,
        vlayer_mask=vlayer_mask,
        map_canvas=MagicMock(),
        inference_parameters=inference_parameters,
    )

    # just run - we will check the results in a more detailed test
    map_processor.run()


def generic_processing_test__specified_extent_from_active_map_extent():
    qgs = init_qgis()

    rlayer = create_rlayer_from_file(RASTER_FILE_PATH)
    model_wrapper = MagicMock()
    model_wrapper.process = lambda x: x[:, :, 0]
    model_wrapper.get_number_of_channels = lambda: 3

    inference_parameters = InferenceParameters(
        resolution_cm_per_px=3,
        tile_size_px=512,
        processed_area_type=ProcessedAreaType.VISIBLE_PART,
        mask_layer_id=None,
        input_layer_id=rlayer.id(),
        input_channels_mapping=INPUT_CHANNELS_MAPPING,
        postprocessing_dilate_erode_size=5,
        processing_overlap_percentage=20,
        model=model_wrapper,
    )
    processed_extent = PROCESSED_EXTENT_1

    # we want to use a fake extent, which is the Visible Part of the map,
    # so we need to mock its function calls
    inference_parameters.processed_area_type = ProcessedAreaType.VISIBLE_PART
    map_canvas = MagicMock()
    map_canvas.extent = lambda: processed_extent
    map_canvas.mapSettings().destinationCrs = lambda: QgsCoordinateReferenceSystem("EPSG:32633")

    map_processor = MapProcessor(
        rlayer=rlayer,
        vlayer_mask=None,
        map_canvas=map_canvas,
        inference_parameters=inference_parameters,
    )

    # just run - we will check the results in a more detailed test
    map_processor.run()


if __name__ == '__main__':
    dummy_model_processing__entire_file()
    generic_processing_test__specified_extent_from_vlayer()
    generic_processing_test__specified_extent_from_active_map_extent()
    print('Done')
