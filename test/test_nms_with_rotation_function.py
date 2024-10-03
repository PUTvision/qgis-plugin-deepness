from unittest.mock import MagicMock
import numpy as np

from deepness.processing.models.detector import Detection, Detector
from deepness.processing.processing_utils import BoundingBox
from test.test_utils import init_qgis


def test_nms_with_rotation_function():
    detections = [
        Detection(bbox=BoundingBox(x_min=90, x_max=173, y_min=338, y_max=422, rot=0), conf=0.9426471, clss=0),
        Detection(bbox=BoundingBox(x_min=297, x_max=380, y_min=340, y_max=426, rot=np.pi/2), conf=0.9370736, clss=0),
        Detection(bbox=BoundingBox(x_min=481, x_max=512, y_min=404, y_max=501, rot=-np.pi/2), conf=0.88433576, clss=0),
        Detection(bbox=BoundingBox(x_min=479, x_max=563, y_min=411, y_max=497, rot=0), conf=0.94546044, clss=0),
        Detection(bbox=BoundingBox(x_min=307, x_max=381, y_min=341, y_max=427, rot=np.pi/2), conf=0.94528174, clss=0),
        Detection(bbox=BoundingBox(x_min=659, x_max=741, y_min=484, y_max=512, rot=-np.pi/2), conf=0.8914648, clss=0),
        Detection(bbox=BoundingBox(x_min=657, x_max=741, y_min=482, y_max=512, rot=0), conf=0.86605114, clss=0),
        Detection(bbox=BoundingBox(x_min=90, x_max=173, y_min=339, y_max=422, rot=np.pi/2), conf=0.9531542, clss=0),
        Detection(bbox=BoundingBox(x_min=225, x_max=308, y_min=533, y_max=616, rot=-np.pi/2), conf=0.94643426, clss=0),
        Detection(bbox=BoundingBox(x_min=297, x_max=380, y_min=340, y_max=427, rot=0), conf=0.9433237, clss=0),
        Detection(bbox=BoundingBox(x_min=406, x_max=490, y_min=602, y_max=688, rot=0), conf=0.93923676, clss=0),
        Detection(bbox=BoundingBox(x_min=480, x_max=512, y_min=406, y_max=502, rot=0), conf=0.82953537, clss=0),
        Detection(bbox=BoundingBox(x_min=587, x_max=675, y_min=670, y_max=761, rot=0), conf=0.9486014, clss=0),
        Detection(bbox=BoundingBox(x_min=308, x_max=381, y_min=339, y_max=427, rot=0), conf=0.94426584, clss=0),
        Detection(bbox=BoundingBox(x_min=407, x_max=490, y_min=604, y_max=689, rot=0), conf=0.94282913, clss=0),
        Detection(bbox=BoundingBox(x_min=660, x_max=744, y_min=481, y_max=567, rot=0), conf=0.942073, clss=0),
        Detection(bbox=BoundingBox(x_min=479, x_max=563, y_min=411, y_max=498, rot=0), conf=0.9392239, clss=0),
        Detection(bbox=BoundingBox(x_min=781, x_max=820, y_min=729, y_max=817, rot=0), conf=0.9145942, clss=0),
        Detection(bbox=BoundingBox(x_min=534, x_max=613, y_min=802, y_max=819, rot=0), conf=0.67225057, clss=0),
        Detection(bbox=BoundingBox(x_min=850, x_max=932, y_min=546, y_max=627, rot=0), conf=0.9545207, clss=0),
        Detection(bbox=BoundingBox(x_min=780, x_max=862, y_min=733, y_max=813, rot=0), conf=0.947471, clss=0),
        Detection(bbox=BoundingBox(x_min=659, x_max=744, y_min=480, y_max=567, rot=0), conf=0.9352352, clss=0),
        Detection(bbox=BoundingBox(x_min=616, x_max=673, y_min=667, y_max=760, rot=0), conf=0.88940847, clss=0),
    ]

    bboxes = []
    confs = []
    for detection in detections:
        bboxes.append(detection.bbox.get_xyxy_rot())
        confs.append(detection.conf)

    picks = Detector.non_max_suppression_fast(np.array(bboxes), np.array(confs), 0.2, with_rot=True)

    detections = [d for i, d in enumerate(detections) if i in picks]

    assert len(detections) == 11

    inter = []
    for i in range(len(detections)):
        for j in range(i+1, len(detections)):
            if i != j:
                inter.append(detections[i].bbox.calculate_intersection_over_smaler_area(detections[j].bbox))

    assert np.all(np.array(inter) < 0.2)

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

if __name__ == '__main__':
    test_nms_with_rotation_function()
    test_nms_human_case_with_rotation()

