from test.test_utils import (create_default_input_channels_mapping_for_rgba_bands, create_rlayer_from_file,
                             create_vlayer_from_file, get_dummy_fotomap_area_crs3857_path, get_dummy_fotomap_area_path,
                             get_dummy_fotomap_small_path, get_dummy_segmentation_model_path,
                             get_dummy_sigmoid_model_path, init_qgis)
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

VLAYER_MASK_FILE_PATH = get_dummy_fotomap_area_path()

VLAYER_MASK_CRS3857_FILE_PATH = get_dummy_fotomap_area_crs3857_path()

MODEL_FILE_PATH = get_dummy_sigmoid_model_path()

INPUT_CHANNELS_MAPPING = create_default_input_channels_mapping_for_rgba_bands()


def test_sigmoid_model_processing__entire_file():
    qgs = init_qgis()

    rlayer = create_rlayer_from_file(RASTER_FILE_PATH)
    model = Segmentor(MODEL_FILE_PATH)

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
        pixel_classification__probability_threshold=0.6,
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
    pixels = np.unique(result_img, return_counts=True)
    assert abs(pixels[1][1] - 25002) < 50  # number of RED pixels in the image

    # you should see only the part of RASTER_FILE_PATH that is pure red pixels. Use snippet below for debugging
    # from matplotlib import pyplot as plt
    # plt.imshow(result_img[0])
    # plt.show()

    # TODO - add detailed check for pixel values once we have output channels mapping with thresholding


if __name__ == '__main__':
    test_sigmoid_model_processing__entire_file()
    print('Done')
