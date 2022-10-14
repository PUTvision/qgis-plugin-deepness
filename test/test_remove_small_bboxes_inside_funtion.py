from unittest.mock import MagicMock
import numpy as np

from deepness.processing.models.detector import Detection, Detector
from deepness.processing.processing_utils import BoundingBox
from test.test_utils import init_qgis


def test_remove_small_bboxes_inside_funtion():
    detections = [
        Detection(bbox=BoundingBox(x_min=0, x_max=100, y_min=0, y_max=100), conf=0.99999, clss=0),
        Detection(bbox=BoundingBox(x_min=10, x_max=20, y_min=10, y_max=20), conf=0.88888, clss=0),
        Detection(bbox=BoundingBox(x_min=100, x_max=120, y_min=0, y_max=1000), conf=0.88888, clss=0),
    ]

    filtered_bounding_boxes = sorted(detections, reverse=True)

    to_remove = []
    for i in range(len(filtered_bounding_boxes)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(filtered_bounding_boxes)):
            if j in to_remove:
                continue
            if i != j:
                if filtered_bounding_boxes[i].bbox.calculate_intersection_over_smaler_area(
                        filtered_bounding_boxes[j].bbox) > 0.2:
                    to_remove.append(j)

    filtered_bounding_boxes = [x for i, x in enumerate(filtered_bounding_boxes) if i not in to_remove]

    assert len(filtered_bounding_boxes) == 2

if __name__ == '__main__':
    test_remove_small_bboxes_inside_funtion()

