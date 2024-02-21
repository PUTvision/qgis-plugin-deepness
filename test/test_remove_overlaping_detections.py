from test.test_utils import get_predicted_detections_path, init_qgis

import numpy as np

from deepness.processing.map_processor.map_processor_detection import MapProcessorDetection
from deepness.processing.models.detector import Detection
from deepness.processing.processing_utils import BoundingBox


def test__remove_overlaping_detections():
    init_qgis()

    with open(get_predicted_detections_path(), 'rb') as f:
        dets = np.load(f)

    detections = []
    for d in dets:
        detections.append(Detection(bbox=BoundingBox(x_min=d[0], y_min=d[1], x_max=d[2], y_max=d[3]), conf=d[4], clss=0))

    returns = MapProcessorDetection.remove_overlaping_detections(detections, 0.5)

    assert len(detections) == 3254
    assert len(returns) == 2430


if __name__ == '__main__':
    test__remove_overlaping_detections()
