from unittest.mock import MagicMock

from qgis.core import QgsCoordinateReferenceSystem, QgsRectangle

from deep_segmentation_framework.common.processing_parameters.segmentation_parameters import SegmentationParameters
from deep_segmentation_framework.common.processing_parameters.map_processing_parameters import ProcessedAreaType, \
    ModelOutputFormat
from deep_segmentation_framework.processing.map_processor.map_processor_segmentation import MapProcessorSegmentation
from deep_segmentation_framework.processing.models.segmentor import Segmentor
from test.test_utils import init_qgis, create_rlayer_from_file, \
    create_vlayer_from_file, get_dummy_fotomap_area_path, get_dummy_fotomap_small_path, get_dummy_segmentation_model_path, \
    create_default_input_channels_mapping_for_rgba_bands

import numpy as np


RASTER_FILE_PATH = get_dummy_fotomap_small_path()

VLAYER_MASK_FILE_PATH = get_dummy_fotomap_area_path()

MODEL_FILE_PATH = get_dummy_segmentation_model_path()

INPUT_CHANNELS_MAPPING = create_default_input_channels_mapping_for_rgba_bands()

PROCESSED_EXTENT_1 = QgsRectangle(  # big part of the fotomap
        638840.370, 5802593.197,
        638857.695, 5802601.792)


def test_dummy_model_processing__entire_file():
    qgs = init_qgis()

    rlayer = create_rlayer_from_file(RASTER_FILE_PATH)
    model = Segmentor(MODEL_FILE_PATH)

    params = SegmentationParameters(
        resolution_cm_per_px=3,
        tile_size_px=model.get_input_size_in_pixels()[0],  # same x and y dimensions, so take x
        processed_area_type=ProcessedAreaType.ENTIRE_LAYER,
        mask_layer_id=None,
        input_layer_id=rlayer.id(),
        input_channels_mapping=INPUT_CHANNELS_MAPPING,
        postprocessing_dilate_erode_size=5,
        processing_overlap_percentage=20,
        pixel_classification__probability_threshold=0.5,
        model_output_format=ModelOutputFormat.ALL_CLASSES_AS_SEPARATE_LAYERS,
        model_output_format__single_class_number=-1,
        model=model,
    )

    map_processor = MapProcessorSegmentation(
        rlayer=rlayer,
        vlayer_mask=None,
        map_canvas=MagicMock(),
        params=params,
    )

    map_processor.run()
    result_img = map_processor.get_result_img()

    assert result_img.shape == (561, 829)
    # TODO - add detailed check for pixel values once we have output channels mapping with thresholding


def model_process_mock(x):
    x = x[:, :, 0:2]
    return np.transpose(x, (2, 0, 1))


def test_generic_processing_test__specified_extent_from_vlayer():
    qgs = init_qgis()

    rlayer = create_rlayer_from_file(RASTER_FILE_PATH)
    vlayer_mask = create_vlayer_from_file(VLAYER_MASK_FILE_PATH)
    vlayer_mask.setCrs(rlayer.crs())
    model = MagicMock()
    model.process = model_process_mock
    model.get_number_of_channels = lambda: 2
    model.get_number_of_output_channels = lambda: 2
    model.get_channel_name = lambda x: str(x)

    params = SegmentationParameters(
        resolution_cm_per_px=3,
        tile_size_px=512,
        processed_area_type=ProcessedAreaType.FROM_POLYGONS,
        mask_layer_id=vlayer_mask.id(),
        input_layer_id=rlayer.id(),
        input_channels_mapping=INPUT_CHANNELS_MAPPING,
        postprocessing_dilate_erode_size=5,
        processing_overlap_percentage=20,
        pixel_classification__probability_threshold=0.5,
        model_output_format=ModelOutputFormat.ONLY_SINGLE_CLASS_AS_LAYER,
        model_output_format__single_class_number=1,
        model=model,
    )
    map_processor = MapProcessorSegmentation(
        rlayer=rlayer,
        vlayer_mask=vlayer_mask,
        map_canvas=MagicMock(),
        params=params,
    )

    # just run - we will check the results in a more detailed test
    map_processor.run()


def test_generic_processing_test__specified_extent_from_active_map_extent():
    qgs = init_qgis()

    rlayer = create_rlayer_from_file(RASTER_FILE_PATH)
    model = MagicMock()
    model.process = model_process_mock
    model.get_number_of_channels = lambda: 2
    model.get_number_of_output_channels = lambda: 2
    model.get_channel_name = lambda x: str(x)

    params = SegmentationParameters(
        resolution_cm_per_px=3,
        tile_size_px=512,
        processed_area_type=ProcessedAreaType.VISIBLE_PART,
        mask_layer_id=None,
        input_layer_id=rlayer.id(),
        input_channels_mapping=INPUT_CHANNELS_MAPPING,
        postprocessing_dilate_erode_size=5,
        processing_overlap_percentage=20,
        pixel_classification__probability_threshold=0.5,
        model_output_format=ModelOutputFormat.CLASSES_AS_SEPARATE_LAYERS_WITHOUT_ZERO_CLASS,
        model_output_format__single_class_number=-1,
        model=model,
    )
    processed_extent = PROCESSED_EXTENT_1

    # we want to use a fake extent, which is the Visible Part of the map,
    # so we need to mock its function calls
    params.processed_area_type = ProcessedAreaType.VISIBLE_PART
    map_canvas = MagicMock()
    map_canvas.extent = lambda: processed_extent
    map_canvas.mapSettings().destinationCrs = lambda: QgsCoordinateReferenceSystem("EPSG:32633")

    map_processor = MapProcessorSegmentation(
        rlayer=rlayer,
        vlayer_mask=None,
        map_canvas=map_canvas,
        params=params,
    )

    # just run - we will check the results in a more detailed test
    map_processor.run()


if __name__ == '__main__':
    test_dummy_model_processing__entire_file()
    test_generic_processing_test__specified_extent_from_vlayer()
    test_generic_processing_test__specified_extent_from_active_map_extent()
    print('Done')
