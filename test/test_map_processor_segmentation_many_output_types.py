from test.test_utils import (create_default_input_channels_mapping_for_rgba_bands, create_rlayer_from_file,
                             get_dummy_fotomap_small_path, get_dummy_segmentation_models_dict, init_qgis)
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
from qgis.core import QgsCoordinateReferenceSystem, QgsRectangle

from deepness.common.processing_overlap import ProcessingOverlap, ProcessingOverlapOptions
from deepness.common.processing_parameters.map_processing_parameters import ProcessedAreaType
from deepness.common.processing_parameters.segmentation_parameters import SegmentationParameters
from deepness.processing.map_processor.map_processor_segmentation import MapProcessorSegmentation
from deepness.processing.models.segmentor import Segmentor

RASTER_FILE_PATH = get_dummy_fotomap_small_path()
INPUT_CHANNELS_MAPPING = create_default_input_channels_mapping_for_rgba_bands()

MODEL_FILES_DICT = get_dummy_segmentation_models_dict()

# 'one_output': {
#             '1x1x512x512'
#             '1x512x512'
#             '1x2x512x512'
#         },
#         'two_outputs': {
#             '1x1x512x512'
#             '1x512x512'
#             '1x2x512x512'
#         }


def test_dummy_model_segmentation_processing__1x1x512x512():
    qgs = init_qgis()

    rlayer = create_rlayer_from_file(RASTER_FILE_PATH)
    model = Segmentor(MODEL_FILES_DICT['one_output']['1x1x512x512'])

    params = SegmentationParameters(
        resolution_cm_per_px=3,
        tile_size_px=model.get_input_size_in_pixels()[0],  # same x and y dimensions, so take x
        batch_size=1,
        local_cache=False,
        processed_area_type=ProcessedAreaType.ENTIRE_LAYER,
        mask_layer_id=None,
        input_layer_id=rlayer.id(),
        input_channels_mapping=INPUT_CHANNELS_MAPPING,
        postprocessing_dilate_erode_size=5,
        processing_overlap=ProcessingOverlap(ProcessingOverlapOptions.OVERLAP_IN_PERCENT, percentage=20),
        pixel_classification__probability_threshold=0.5,
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

    assert result_img.shape == (1, 561, 829)
    
    channels = map_processor._get_indexes_of_model_output_channels_to_create()
    assert len(channels) == 1
    assert channels[0] == 1
    
    name = map_processor.model.get_channel_name(0, 0)
    assert name == 'Coffee'

def test_dummy_model_segmentation_processing__1x512x512():
    qgs = init_qgis()

    rlayer = create_rlayer_from_file(RASTER_FILE_PATH)
    model = Segmentor(MODEL_FILES_DICT['one_output']['1x512x512'])

    params = SegmentationParameters(
        resolution_cm_per_px=3,
        tile_size_px=model.get_input_size_in_pixels()[0],  # same x and y dimensions, so take x
        batch_size=1,
        local_cache=False,
        processed_area_type=ProcessedAreaType.ENTIRE_LAYER,
        mask_layer_id=None,
        input_layer_id=rlayer.id(),
        input_channels_mapping=INPUT_CHANNELS_MAPPING,
        postprocessing_dilate_erode_size=5,
        processing_overlap=ProcessingOverlap(ProcessingOverlapOptions.OVERLAP_IN_PERCENT, percentage=20),
        pixel_classification__probability_threshold=0.5,
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

    assert result_img.shape == (1, 561, 829)
    
    channels = map_processor._get_indexes_of_model_output_channels_to_create()
    assert len(channels) == 1
    assert channels[0] == 1
    
    name = map_processor.model.get_channel_name(0, 0)
    assert name == 'Coffee'

def test_dummy_model_segmentation_processing__1x2x512x512():
    qgs = init_qgis()

    rlayer = create_rlayer_from_file(RASTER_FILE_PATH)
    model = Segmentor(MODEL_FILES_DICT['one_output']['1x2x512x512'])

    params = SegmentationParameters(
        resolution_cm_per_px=3,
        tile_size_px=model.get_input_size_in_pixels()[0],  # same x and y dimensions, so take x
        batch_size=1,
        local_cache=False,
        processed_area_type=ProcessedAreaType.ENTIRE_LAYER,
        mask_layer_id=None,
        input_layer_id=rlayer.id(),
        input_channels_mapping=INPUT_CHANNELS_MAPPING,
        postprocessing_dilate_erode_size=5,
        processing_overlap=ProcessingOverlap(ProcessingOverlapOptions.OVERLAP_IN_PERCENT, percentage=20),
        pixel_classification__probability_threshold=0.5,
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

    assert result_img.shape == (1, 561, 829)

# two outputs

def test_dummy_model_segmentation_processing__two_outputs_1x1x512x512():
    qgs = init_qgis()

    rlayer = create_rlayer_from_file(RASTER_FILE_PATH)
    model = Segmentor(MODEL_FILES_DICT['two_outputs']['1x1x512x512'])

    params = SegmentationParameters(
        resolution_cm_per_px=3,
        tile_size_px=model.get_input_size_in_pixels()[0],  # same x and y dimensions, so take x
        batch_size=1,
        local_cache=False,
        processed_area_type=ProcessedAreaType.ENTIRE_LAYER,
        mask_layer_id=None,
        input_layer_id=rlayer.id(),
        input_channels_mapping=INPUT_CHANNELS_MAPPING,
        postprocessing_dilate_erode_size=5,
        processing_overlap=ProcessingOverlap(ProcessingOverlapOptions.OVERLAP_IN_PERCENT, percentage=20),
        pixel_classification__probability_threshold=0.5,
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

    assert result_img.shape == (2, 561, 829)
    
    channels = map_processor._get_indexes_of_model_output_channels_to_create()
    assert len(channels) == 2
    assert channels[0] == 1
    assert channels[1] == 1
    
    name = map_processor.model.get_channel_name(0, 0)
    assert name == 'Coffee'
    
    name = map_processor.model.get_channel_name(1, 0)
    assert name == 'Juice'

def test_dummy_model_segmentation_processing__two_outputs_1x512x512():
    qgs = init_qgis()

    rlayer = create_rlayer_from_file(RASTER_FILE_PATH)
    model = Segmentor(MODEL_FILES_DICT['two_outputs']['1x512x512'])

    params = SegmentationParameters(
        resolution_cm_per_px=3,
        tile_size_px=model.get_input_size_in_pixels()[0],  # same x and y dimensions, so take x
        batch_size=1,
        local_cache=False,
        processed_area_type=ProcessedAreaType.ENTIRE_LAYER,
        mask_layer_id=None,
        input_layer_id=rlayer.id(),
        input_channels_mapping=INPUT_CHANNELS_MAPPING,
        postprocessing_dilate_erode_size=5,
        processing_overlap=ProcessingOverlap(ProcessingOverlapOptions.OVERLAP_IN_PERCENT, percentage=20),
        pixel_classification__probability_threshold=0.5,
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

    assert result_img.shape == (2, 561, 829)



def test_dummy_model_segmentation_processing__two_outputs_1x2x512x512():
    qgs = init_qgis()

    rlayer = create_rlayer_from_file(RASTER_FILE_PATH)
    model = Segmentor(MODEL_FILES_DICT['two_outputs']['1x2x512x512'])

    params = SegmentationParameters(
        resolution_cm_per_px=3,
        tile_size_px=model.get_input_size_in_pixels()[0],  # same x and y dimensions, so take x
        batch_size=1,
        local_cache=False,
        processed_area_type=ProcessedAreaType.ENTIRE_LAYER,
        mask_layer_id=None,
        input_layer_id=rlayer.id(),
        input_channels_mapping=INPUT_CHANNELS_MAPPING,
        postprocessing_dilate_erode_size=5,
        processing_overlap=ProcessingOverlap(ProcessingOverlapOptions.OVERLAP_IN_PERCENT, percentage=20),
        pixel_classification__probability_threshold=0.5,
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

    assert result_img.shape == (2, 561, 829)
    
    channels = map_processor._get_indexes_of_model_output_channels_to_create()
    assert len(channels) == 2
    assert channels[0] == 2
    assert channels[1] == 2
    
    name = map_processor.model.get_channel_name(0, 0)
    assert name == 'Coffee'
    
    name = map_processor.model.get_channel_name(0, 1)
    assert name == 'Tea'
    
    name = map_processor.model.get_channel_name(1, 0)
    assert name == 'Juice'
    
    name = map_processor.model.get_channel_name(1, 1)
    assert name == 'Beer'

if __name__ == '__main__':
    test_dummy_model_segmentation_processing__1x1x512x512()
    test_dummy_model_segmentation_processing__1x512x512()
    test_dummy_model_segmentation_processing__1x2x512x512()
    
    test_dummy_model_segmentation_processing__two_outputs_1x1x512x512()
    test_dummy_model_segmentation_processing__two_outputs_1x512x512()
    test_dummy_model_segmentation_processing__two_outputs_1x2x512x512()
    print('All tests passed')
