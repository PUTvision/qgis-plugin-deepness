from test.test_utils import (create_default_input_channels_mapping_for_rgba_bands, create_rlayer_from_file,
                             get_dummy_fotomap_small_path, get_dummy_regression_models_dict, init_qgis)
from unittest.mock import MagicMock

import numpy as np
from qgis.core import QgsCoordinateReferenceSystem, QgsRectangle

from deepness.common.processing_overlap import ProcessingOverlap, ProcessingOverlapOptions
from deepness.common.processing_parameters.map_processing_parameters import ProcessedAreaType
from deepness.common.processing_parameters.regression_parameters import RegressionParameters
from deepness.processing.map_processor.map_processor_regression import MapProcessorRegression
from deepness.processing.models.regressor import Regressor

RASTER_FILE_PATH = get_dummy_fotomap_small_path()
INPUT_CHANNELS_MAPPING = create_default_input_channels_mapping_for_rgba_bands()


MODEL_FILES_DICT = get_dummy_regression_models_dict()

# 'one_output': {
#             '1x1x512x512'
#             '1x512x512'
#         },
#         'two_outputs': {
#             '1x1x512x512'
#             '1x512x512'
#         }


def test_dummy_model_regression_processing__1x512x512():
    qgs = init_qgis()

    rlayer = create_rlayer_from_file(RASTER_FILE_PATH)
    model = Regressor(MODEL_FILES_DICT['one_output']['1x512x512'])

    params = RegressionParameters(
        resolution_cm_per_px=3,
        tile_size_px=model.get_input_size_in_pixels()[0],  # same x and y dimensions, so take x
        batch_size=1,
        local_cache=False,
        processed_area_type=ProcessedAreaType.ENTIRE_LAYER,
        mask_layer_id=None,
        input_layer_id=rlayer.id(),
        input_channels_mapping=INPUT_CHANNELS_MAPPING,
        output_scaling=1.0,
        processing_overlap=ProcessingOverlap(ProcessingOverlapOptions.OVERLAP_IN_PERCENT, percentage=20),
        model=model,
    )

    map_processor = MapProcessorRegression(
        rlayer=rlayer,
        vlayer_mask=None,
        map_canvas=MagicMock(),
        params=params,
    )

    map_processor.run()
    result_imgs = map_processor.get_result_img()

    assert result_imgs.shape == (1, 561, 829)

def test_dummy_model_regression_processing__1x1x512x512():
    qgs = init_qgis()

    rlayer = create_rlayer_from_file(RASTER_FILE_PATH)
    model = Regressor(MODEL_FILES_DICT['one_output']['1x1x512x512'])

    params = RegressionParameters(
        resolution_cm_per_px=3,
        tile_size_px=model.get_input_size_in_pixels()[0],  # same x and y dimensions, so take x
        batch_size=1,
        local_cache=False,
        processed_area_type=ProcessedAreaType.ENTIRE_LAYER,
        mask_layer_id=None,
        input_layer_id=rlayer.id(),
        input_channels_mapping=INPUT_CHANNELS_MAPPING,
        output_scaling=1.0,
        processing_overlap=ProcessingOverlap(ProcessingOverlapOptions.OVERLAP_IN_PERCENT, percentage=20),
        model=model,
    )

    map_processor = MapProcessorRegression(
        rlayer=rlayer,
        vlayer_mask=None,
        map_canvas=MagicMock(),
        params=params,
    )

    map_processor.run()
    result_imgs = map_processor.get_result_img()

    assert result_imgs.shape == (1, 561, 829)

def test_dummy_model_regression_processing__1x512x512():
    qgs = init_qgis()

    rlayer = create_rlayer_from_file(RASTER_FILE_PATH)
    model = Regressor(MODEL_FILES_DICT['two_outputs']['1x512x512'])

    params = RegressionParameters(
        resolution_cm_per_px=3,
        tile_size_px=model.get_input_size_in_pixels()[0],  # same x and y dimensions, so take x
        batch_size=1,
        local_cache=False,
        processed_area_type=ProcessedAreaType.ENTIRE_LAYER,
        mask_layer_id=None,
        input_layer_id=rlayer.id(),
        input_channels_mapping=INPUT_CHANNELS_MAPPING,
        output_scaling=1.0,
        processing_overlap=ProcessingOverlap(ProcessingOverlapOptions.OVERLAP_IN_PERCENT, percentage=20),
        model=model,
    )

    map_processor = MapProcessorRegression(
        rlayer=rlayer,
        vlayer_mask=None,
        map_canvas=MagicMock(),
        params=params,
    )

    map_processor.run()
    result_imgs = map_processor.get_result_img()

    assert result_imgs.shape == (2, 561, 829)

def test_dummy_model_regression_processing__1x1x512x512():
    qgs = init_qgis()

    rlayer = create_rlayer_from_file(RASTER_FILE_PATH)
    model = Regressor(MODEL_FILES_DICT['two_outputs']['1x1x512x512'])

    params = RegressionParameters(
        resolution_cm_per_px=3,
        tile_size_px=model.get_input_size_in_pixels()[0],  # same x and y dimensions, so take x
        batch_size=1,
        local_cache=False,
        processed_area_type=ProcessedAreaType.ENTIRE_LAYER,
        mask_layer_id=None,
        input_layer_id=rlayer.id(),
        input_channels_mapping=INPUT_CHANNELS_MAPPING,
        output_scaling=1.0,
        processing_overlap=ProcessingOverlap(ProcessingOverlapOptions.OVERLAP_IN_PERCENT, percentage=20),
        model=model,
    )

    map_processor = MapProcessorRegression(
        rlayer=rlayer,
        vlayer_mask=None,
        map_canvas=MagicMock(),
        params=params,
    )

    map_processor.run()
    result_imgs = map_processor.get_result_img()

    assert result_imgs.shape == (2, 561, 829)

if __name__ == '__init__':
    test_dummy_model_regression_processing__1x512x512()
    test_dummy_model_regression_processing__1x1x512x512()
    test_dummy_model_regression_processing__1x512x512()
    test_dummy_model_regression_processing__1x1x512x512()
    print('All tests passed!')
