import os
from pathlib import Path
from test.test_utils import create_default_input_channels_mapping_for_rgb_bands, create_rlayer_from_file, init_qgis
from unittest.mock import MagicMock

from deepness.common.processing_overlap import ProcessingOverlap, ProcessingOverlapOptions
from deepness.common.processing_parameters.map_processing_parameters import ModelOutputFormat, ProcessedAreaType
from deepness.common.processing_parameters.segmentation_parameters import SegmentationParameters
from deepness.processing.map_processor.map_processor_segmentation import MapProcessorSegmentation
from deepness.processing.models.segmentor import Segmentor

HOME_DIR = Path(__file__).resolve().parents[1]
EXAMPLE_DATA_DIR = os.path.join(HOME_DIR, 'examples', 'deeplabv3_segmentation_landcover')

MODEL_FILE_PATH = os.path.join(EXAMPLE_DATA_DIR, 'deeplabv3_landcover_4c.onnx')
RASTER_FILE_PATH = os.path.join(EXAMPLE_DATA_DIR, 'N-33-60-D-c-4-2.tif')

INPUT_CHANNELS_MAPPING = create_default_input_channels_mapping_for_rgb_bands()


def test_map_processor_segmentation_landcover_example():
    qgs = init_qgis()

    rlayer = create_rlayer_from_file(RASTER_FILE_PATH)
    model = Segmentor(MODEL_FILE_PATH)

    params = SegmentationParameters(
        resolution_cm_per_px=100,
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

    assert result_img.shape == (2351, 2068)


if __name__ == '__main__':
    test_map_processor_segmentation_landcover_example()
    print('Done')
