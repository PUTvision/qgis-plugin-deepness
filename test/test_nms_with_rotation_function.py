from unittest.mock import MagicMock
import numpy as np

from deepness.processing.models.detector import Detection, Detector
from deepness.processing.processing_utils import BoundingBox
from test.test_utils import init_qgis


def test_nms_human_case_with_rotation():
    detections = [
        Detection(bbox=BoundingBox(x_min=10, x_max=80, y_min=20, y_max=150, rot=0), conf=0.99999, clss=0),
        Detection(bbox=BoundingBox(x_min=0, x_max=90, y_min=100, y_max=190, rot=0), conf=0.44444, clss=0),
        Detection(bbox=BoundingBox(x_min=0, x_max=100, y_min=0, y_max=200, rot=0), conf=0.88888, clss=0),
    ]

    bboxes = []
    confs = []
    for detection in detections:
        bboxes.append(detection.bbox.get_xyxy_rot())
        confs.append(detection.conf)

    picks = Detector.non_max_suppression_fast(np.array(bboxes), np.array(confs), 0.2, with_rot=True)

    detections = [d for i, d in enumerate(detections) if i in picks]
    assert len(detections) == 1


def test_nms_human_case_with_rotation_v2():
    detections = [
        Detection(bbox=BoundingBox(x_min=100, x_max=200, y_min=100, y_max=200, rot=0), conf=0.99999, clss=0),
        Detection(bbox=BoundingBox(x_min=100, x_max=200, y_min=100, y_max=200, rot=np.pi/2), conf=0.99999, clss=0),
        Detection(bbox=BoundingBox(x_min=100, x_max=200, y_min=100, y_max=200, rot=-np.pi/2), conf=0.99999, clss=0),
        Detection(bbox=BoundingBox(x_min=100, x_max=200, y_min=100, y_max=200, rot=np.pi), conf=0.99999, clss=0),
    ]

    bboxes = []
    confs = []
    for detection in detections:
        bboxes.append(detection.bbox.get_xyxy_rot())
        confs.append(detection.conf)

    picks = Detector.non_max_suppression_fast(np.array(bboxes), np.array(confs), 0.2, with_rot=True)

    detections = [d for i, d in enumerate(detections) if i in picks]
    assert len(detections) == 1

if __name__ == '__main__':
    test_nms_human_case_with_rotation()
    test_nms_human_case_with_rotation_v2()

