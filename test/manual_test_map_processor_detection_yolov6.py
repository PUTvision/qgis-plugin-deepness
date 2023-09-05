import os
from pathlib import Path
from test.test_utils import create_default_input_channels_mapping_for_rgb_bands, create_rlayer_from_file, init_qgis
from unittest.mock import MagicMock

from deepness.common.processing_parameters.detection_parameters import DetectionParameters, DetectorType
from deepness.common.processing_parameters.map_processing_parameters import ModelOutputFormat, ProcessedAreaType
from deepness.processing.map_processor.map_processor_detection import MapProcessorDetection
from deepness.processing.models.detector import Detector

# Files and model from github issue: https://github.com/PUTvision/qgis-plugin-deepness/discussions/101

HOME_DIR = Path(__file__).resolve().parents[1]
EXAMPLE_DATA_DIR = os.path.join(HOME_DIR, 'examples', 'manually_downloaded')

MODEL_FILE_PATH = os.path.join(EXAMPLE_DATA_DIR, 'best_ckpt.onnx')
RASTER_FILE_PATH = os.path.join(EXAMPLE_DATA_DIR, 'map_N.tif')

INPUT_CHANNELS_MAPPING = create_default_input_channels_mapping_for_rgb_bands()


def test_map_processor_detection_yolov6():
    qgs = init_qgis()

    rlayer = create_rlayer_from_file(RASTER_FILE_PATH)
    model_wrapper = Detector(MODEL_FILE_PATH)

    params = DetectionParameters(
        resolution_cm_per_px=2,
        tile_size_px=model_wrapper.get_input_size_in_pixels()[0],  # same x and y dimensions, so take x
        processed_area_type=ProcessedAreaType.ENTIRE_LAYER,
        mask_layer_id=None,
        input_layer_id=rlayer.id(),
        input_channels_mapping=INPUT_CHANNELS_MAPPING,
        processing_overlap_percentage=0,
        model=model_wrapper,
        confidence=0.9,
        iou_threshold=0.4,
        remove_overlapping_detections=False,
        model_output_format=ModelOutputFormat.ALL_CLASSES_AS_SEPARATE_LAYERS,
        model_output_format__single_class_number=-1,
        detector_type=DetectorType.YOLO_v6,
    )

    map_processor = MapProcessorDetection(
        rlayer=rlayer,
        vlayer_mask=None,
        map_canvas=MagicMock(),
        params=params,
    )

    map_processor.run()

    assert len(map_processor.get_all_detections()) == 36


if __name__ == '__main__':
    test_map_processor_detection_yolov6()
    print('Done')
