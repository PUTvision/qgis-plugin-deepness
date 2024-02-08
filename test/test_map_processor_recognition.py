from test.test_utils import (create_default_input_channels_mapping_for_rgb_bands, create_rlayer_from_file,
                             get_dummy_fotomap_area_crs3857_path, get_dummy_fotomap_area_path,
                             get_dummy_recognition_image_path, get_dummy_recognition_map_path,
                             get_dummy_recognition_model_path, init_qgis)
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
from qgis.core import QgsCoordinateReferenceSystem, QgsRectangle

from deepness.common.processing_overlap import ProcessingOverlap, ProcessingOverlapOptions
from deepness.common.processing_parameters.map_processing_parameters import ModelOutputFormat, ProcessedAreaType
from deepness.common.processing_parameters.recognition_parameters import RecognitionParameters
from deepness.processing.map_processor.map_processor_recognition import MapProcessorRecognition
from deepness.processing.models.segmentor import Segmentor

RASTER_FILE_PATH = get_dummy_recognition_map_path()

VLAYER_MASK_FILE_PATH = get_dummy_fotomap_area_path()

VLAYER_MASK_CRS3857_FILE_PATH = get_dummy_fotomap_area_crs3857_path()

MODEL_FILE_PATH = get_dummy_recognition_model_path()
IMAGE_FILE_PATH = get_dummy_recognition_image_path()

INPUT_CHANNELS_MAPPING = create_default_input_channels_mapping_for_rgb_bands()


def test_dummy_model_processing__entire_file():
    qgs = init_qgis()

    rlayer = create_rlayer_from_file(RASTER_FILE_PATH)
    model = Segmentor(MODEL_FILE_PATH)

    params = RecognitionParameters(
        resolution_cm_per_px=50,
        tile_size_px=model.get_input_size_in_pixels()[0],  # same x and y dimensions, so take x
        batch_size=1,
        local_cache=False,
        processed_area_type=ProcessedAreaType.ENTIRE_LAYER,
        mask_layer_id=None,
        input_layer_id=rlayer.id(),
        input_channels_mapping=INPUT_CHANNELS_MAPPING,
        processing_overlap=ProcessingOverlap(ProcessingOverlapOptions.OVERLAP_IN_PERCENT, percentage=0),
        model_output_format=ModelOutputFormat.ALL_CLASSES_AS_SEPARATE_LAYERS,
        model_output_format__single_class_number=-1,
        model=model,
        query_image_path=IMAGE_FILE_PATH,
    )

    map_processor = MapProcessorRecognition(
        rlayer=rlayer,
        vlayer_mask=None,
        map_canvas=MagicMock(),
        params=params,
    )

    map_processor.run()
    result_img = map_processor.get_result_img()

    assert result_img.shape == (1120, 1120)

    vmin = result_img.min()
    vmax = result_img.max()

    assert np.isclose(vmin, 0.9623341, atol=1e-6)
    assert np.isclose(vmax, 1.0, atol=1e-6)

    assert len(np.argwhere(result_img == result_img.max())) == 891
    assert len(np.argwhere(result_img == result_img.min())) == 50176


if __name__ == '__main__':
    test_dummy_model_processing__entire_file()
    print('Done')
