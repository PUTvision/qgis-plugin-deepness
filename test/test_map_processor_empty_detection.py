import os
from pathlib import Path
from test.test_utils import (create_default_input_channels_mapping_for_rgb_bands, create_rlayer_from_file,
                             get_dummy_fotomap_small_path, init_qgis)
from unittest.mock import MagicMock

from deepness.common.processing_overlap import ProcessingOverlap, ProcessingOverlapOptions
from deepness.common.processing_parameters.detection_parameters import DetectionParameters
from deepness.common.processing_parameters.map_processing_parameters import ModelOutputFormat, ProcessedAreaType
from deepness.processing.map_processor.map_processor_detection import MapProcessorDetection
from deepness.processing.models.detector import Detector

HOME_DIR = Path(__file__).resolve().parents[1]
EXAMPLE_DATA_DIR = os.path.join(HOME_DIR, 'examples', 'yolov7_planes_detection_google_earth')

MODEL_FILE_PATH = os.path.join(EXAMPLE_DATA_DIR, 'model_yolov7_tiny_planes_256_1c.onnx')
RASTER_FILE_PATH = get_dummy_fotomap_small_path()

INPUT_CHANNELS_MAPPING = create_default_input_channels_mapping_for_rgb_bands()


def test_map_processor_empty_detection():
    qgs = init_qgis()

    rlayer = create_rlayer_from_file(RASTER_FILE_PATH)
    model_wrapper = Detector(MODEL_FILE_PATH)

    params = DetectionParameters(
        resolution_cm_per_px=100,
        tile_size_px=model_wrapper.get_input_size_in_pixels()[0],  # same x and y dimensions, so take x
        processed_area_type=ProcessedAreaType.ENTIRE_LAYER,
        mask_layer_id=None,
        input_layer_id=rlayer.id(),
        input_channels_mapping=INPUT_CHANNELS_MAPPING,
        processing_overlap=ProcessingOverlap(ProcessingOverlapOptions.OVERLAP_IN_PERCENT, percentage=0),
        model=model_wrapper,
        confidence=0.99,
        iou_threshold=0.99,
        remove_overlapping_detections=False,
        model_output_format=ModelOutputFormat.ALL_CLASSES_AS_SEPARATE_LAYERS,
        model_output_format__single_class_number=-1,
    )

    map_processor = MapProcessorDetection(
        rlayer=rlayer,
        vlayer_mask=None,
        map_canvas=MagicMock(),
        params=params,
    )

    map_processor.run()


if __name__ == '__main__':
    test_map_processor_empty_detection()
    print('Done')
