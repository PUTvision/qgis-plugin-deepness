from dataclasses import dataclass
from typing import Tuple, List

import cv2
import numpy as np

from deepness.processing.models.model_base import ModelBase
from deepness.processing.processing_utils import BoundingBox


@dataclass
class DetectorBBox:
    """
    bounding box - rectangle area, describing position in pixels in detection process
    """
    left_upper: Tuple[int, int]  # left upper corner (x, y)
    right_down: Tuple[int, int]  # bottom right corner (x, y)

    def apply_offset(self, offset_x: int, offset_y: int):
        self.left_upper[0] += offset_x
        self.left_upper[1] += offset_y
        self.right_down[0] += offset_x
        self.right_down[1] += offset_y

    def get_area(self):
        return (self.right_down[0] - self.left_upper[0] + 1) * (self.right_down[1] - self.left_upper[1])

    def get_4_corners(self) -> List[Tuple]:
        """
        Get 4 points (corners) describing the detection rectangle, each point in (x, y) format
        :return:
        """
        return [
            (self.left_upper[0], self.left_upper[1]),
            (self.left_upper[0], self.right_down[1]),
            (self.right_down[0], self.right_down[1]),
            (self.right_down[0], self.left_upper[1]),
        ]

    def get_slice(self):
        """
        Get slice describing bounding box position on an image
        """
        return self.to_bounding_box().get_slice()

    def to_bounding_box(self) -> BoundingBox:
        return BoundingBox(
            x_min=self.left_upper[0],
            x_max=self.right_down[0],
            y_min=self.left_upper[1],
            y_max=self.right_down[1])


@dataclass
class Detection:
    bbox: DetectorBBox  # bounding box describing the detection rectangle
    conf: float  # confidence of the detection
    clss: int  # class of the detected object

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

        self.confidence = None
        self.iou_threshold = None

    def set_inference_params(self, confidence: float, iou_threshold: float):
        self.confidence = confidence
        self.iou_threshold = iou_threshold

    @classmethod
    def get_class_display_name(cls):
        return cls.__name__

    def get_number_of_output_channels(self):
        if len(self.outputs_layers) == 1:
            return self.outputs_layers[0].shape[-1] - 4 - 1 # shape - 4 bboxes - 1 conf
        else:
            return NotImplementedError

    def preprocessing(self, image: np.ndarray):
        img = image[:, :, :self.input_shape[-3]]

        input_data = (img / 255.0)
        input_data = np.transpose(input_data, (2, 0, 1))
        input_batch = np.expand_dims(input_data, 0)
        input_batch = input_batch.astype(np.float32)

        return input_batch

    def postprocessing(self, model_output):
        if self.confidence is None or self.iou_threshold is None:
            return Exception('Confidence or IOU threshold is not set for model. Use self.set_inference_params')

        model_output = model_output[0][0]

        outputs_filtered = np.array(list(filter(lambda x: x[4] >= self.confidence, model_output)))

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
                bbox=DetectorBBox(left_upper=b[:2], right_down=b[2:]),
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

    def check_loaded_model_outputs(self):
        if len(self.outputs_layers) == 1:
            shape = self.outputs_layers[0].shape

            if len(shape) != 3:
                raise Exception(f'Detection model output should have 3 dimensions: (Batch_size, detections, values). '
                                f'Actually has: {shape}')

            if shape[0] != 1:
                raise Exception(f'Detection model can handle only 1-Batch outputs. Has {shape}')
            
        else:
            raise NotImplementedError
