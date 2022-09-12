from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from deep_segmentation_framework.processing.models.model_base import ModelBase


@dataclass
class BBox:
    left_upper: Tuple[int, int]
    right_down: Tuple[int, int]

    def apply_offset(self, offset_x: int, offset_y: int):
        self.left_upper[0] += offset_x
        self.left_upper[1] += offset_y
        self.right_down[0] += offset_x
        self.right_down[1] += offset_y

@dataclass
class Detection:
    bbox: BBox
    conf: float
    clss: int

    def convert_to_global(self, offset_x: int, offset_y: int):
        self.bbox.apply_offset(offset_x=offset_x, offset_y=offset_y)

    def get_bbox_xyxy(self) -> np.ndarray:
        return np.array([
            self.bbox.left_upper[0], self.bbox.left_upper[1],
            self.bbox.right_down[0], self.bbox.right_down[1]]
        )


class Detector(ModelBase):
    def __init__(self, model_file_path: str):
        super(Detector, self).__init__(model_file_path)

        self.score_threshold = 0.6
        self.iou_threshold = 0.5

    @classmethod
    def get_class_display_name(cls):
        return cls.__name__

    def preprocessing(self, image: np.ndarray):
        img = image[:, :, :self.input_shape[-3]]

        input_data = (img / 255.0)
        input_data = np.transpose(input_data, (2, 0, 1))
        input_batch = np.expand_dims(input_data, 0)
        input_batch = input_batch.astype(np.float32)

        return input_batch

    def postprocessing(self, model_output):
        model_output = model_output[0][0]

        outputs_filtered = np.array(list(filter(lambda x: x[4] >= self.score_threshold, model_output)))

        if len(outputs_filtered.shape) < 2:
            return []

        outputs_x1y1x2y2 = self.xywh2xyxy(outputs_filtered)

        pick_indxs = self.non_max_suppression_fast(outputs_x1y1x2y2, self.iou_threshold)
        outputs_nms = outputs_x1y1x2y2[pick_indxs]

        boxes = np.array(outputs_nms[:, :4], dtype=int)
        conf = outputs_nms[:, 4]
        classes = np.argmax(outputs_nms[:, 5:], axis=1)

        detections = []

        for b, c, cl in zip(boxes, conf, classes):
            det = Detection(
                bbox=BBox(left_upper=b[:2], right_down=b[2:]),
                conf=c,
                clss=cl
            )
            detections.append(det)

        return detections

    @staticmethod
    def xywh2xyxy(x):
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    @staticmethod
    def non_max_suppression_fast(boxes: np.ndarray, iou_threshold: float) -> np.ndarray:

        # initialize the list of picked indexes
        pick = []
        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]
            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > iou_threshold)[0])))

        return pick
