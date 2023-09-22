from test.test_utils import (create_default_input_channels_mapping_for_rgba_bands, create_rlayer_from_file,
                             create_vlayer_from_file, get_dummy_fotomap_area_crs3857_path, get_dummy_fotomap_area_path,
                             get_dummy_fotomap_small_path, get_dummy_segmentation_model_different_output_size_path,
                             init_qgis)
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
from qgis.core import QgsCoordinateReferenceSystem, QgsRectangle

from deepness.common.processing_overlap import ProcessingOverlap, ProcessingOverlapOptions
from deepness.common.processing_parameters.map_processing_parameters import ModelOutputFormat, ProcessedAreaType
from deepness.common.processing_parameters.segmentation_parameters import SegmentationParameters
from deepness.processing.map_processor.map_processor_segmentation import MapProcessorSegmentation
from deepness.processing.models.segmentor import Segmentor

RASTER_FILE_PATH = get_dummy_fotomap_small_path()

VLAYER_MASK_FILE_PATH = get_dummy_fotomap_area_path()

VLAYER_MASK_CRS3857_FILE_PATH = get_dummy_fotomap_area_crs3857_path()

MODEL_FILE_PATH = get_dummy_segmentation_model_different_output_size_path()

INPUT_CHANNELS_MAPPING = create_default_input_channels_mapping_for_rgba_bands()

PROCESSED_EXTENT_1 = QgsRectangle(  # big part of the fotomap
    638840.370, 5802593.197,
    638857.695, 5802601.792)


def test_dummy_model_processing_when_different_output_size():
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
        processing_overlap=ProcessingOverlap(ProcessingOverlapOptions.OVERLAP_IN_PERCENT, percentage=20),
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

if __name__ == "__main__":
    test_dummy_model_processing_when_different_output_size()
