""" Module including the class for the object detection task and related functions
"""
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np

from deepness.processing.models.model_base import ModelBase
from deepness.processing.processing_utils import BoundingBox


@dataclass
class Detection:
    """Class that represents single detection result in object detection model

    Parameters
    ----------
    bbox : BoundingBox
        bounding box describing the detection rectangle
    conf : float
        confidence of the detection
    clss : int
        class of the detected object
    """

    bbox: BoundingBox
    """BoundingBox: bounding box describing the detection rectangle"""
    conf: float
    """float: confidence of the detection"""
    clss: int
    """int: class of the detected object"""

    def convert_to_global(self, offset_x: int, offset_y: int):
        """Apply (x,y) offset to bounding box coordinates

        Parameters
        ----------
        offset_x : int
            _description_
        offset_y : int
            _description_
        """
        self.bbox.apply_offset(offset_x=offset_x, offset_y=offset_y)

    def get_bbox_xyxy(self) -> np.ndarray:
        """Convert stored bounding box into x1y1x2y2 format

        Returns
        -------
        np.ndarray
            Array in (x1, y1, x2, y2) format
        """
        return self.bbox.get_xyxy()

    def __lt__(self, other):
        return self.bbox.get_area() < other.bbox.get_area()


class Detector(ModelBase):
    """Class implements object detection features
    
    Detector model is used for detection of objects in images. It is based on YOLOv5/YOLOv7 models style.
    """

    def __init__(self, model_file_path: str):
        """Initialize object detection model
        
        Parameters
        ----------
        model_file_path : str
            Path to model file"""
        super(Detector, self).__init__(model_file_path)

        self.confidence = None
        """float: Confidence threshold"""
        self.iou_threshold = None
        """float: IoU threshold"""

    def set_inference_params(self, confidence: float, iou_threshold: float):
        """Set inference parameters

        Parameters
        ----------
        confidence : float
            Confidence threshold
        iou_threshold : float
            IoU threshold
        """
        self.confidence = confidence
        self.iou_threshold = iou_threshold

    @classmethod
    def get_class_display_name(cls):
        """Get class display name
        
        Returns
        -------
        str
            Class display name"""
        return cls.__name__

    def get_number_of_output_channels(self):
        """Get number of output channels

        Returns
        -------
        int
            Number of output channels            
        """
        if len(self.outputs_layers) == 1:
            return self.outputs_layers[0].shape[-1] - 4 - 1  # shape - 4 bboxes - 1 conf
        else:
            return NotImplementedError

    def preprocessing(self, image: np.ndarray):
        """Preprocess image before inference

        Parameters
        ----------
        image : np.ndarray
            Image to preprocess in RGB format

        Returns
        -------
        np.ndarray
            Preprocessed image
        """
        img = image[:, :, : self.input_shape[-3]]

        input_data = img / 255.0
        input_data = np.transpose(input_data, (2, 0, 1))
        input_batch = np.expand_dims(input_data, 0)
        input_batch = input_batch.astype(np.float32)

        return input_batch

    def postprocessing(self, model_output):
        """Postprocess model output

        Parameters
        ----------
        model_output : list
            Model output
        
        Returns
        -------
        list
            List of detections
        """
        if self.confidence is None or self.iou_threshold is None:
            return Exception(
                "Confidence or IOU threshold is not set for model. Use self.set_inference_params"
            )

        model_output = model_output[0][0]

        outputs_filtered = np.array(
            list(filter(lambda x: x[4] >= self.confidence, model_output))
        )

        if len(outputs_filtered.shape) < 2:
            return []

        outputs_x1y1x2y2 = self.xywh2xyxy(outputs_filtered)

        pick_indxs = self.non_max_suppression_fast(outputs_x1y1x2y2, outputs_filtered[:, 4], self.iou_threshold)
        outputs_nms = outputs_x1y1x2y2[pick_indxs]

        boxes = np.array(outputs_nms[:, :4], dtype=int)
        conf = outputs_nms[:, 4]
        classes = np.argmax(outputs_nms[:, 5:], axis=1)

        detections = []

        for b, c, cl in zip(boxes, conf, classes):
            det = Detection(
                bbox=BoundingBox(
                            x_min = b[0],
                            x_max = b[2],
                            y_min = b[1],
                            y_max = b[3]),
                conf=c,
                clss=cl
            )
            detections.append(det)

        return detections

    @staticmethod
    def xywh2xyxy(x: np.ndarray) -> np.ndarray:
        """Convert bounding box from (x,y,w,h) to (x1,y1,x2,y2) format

        Parameters
        ----------
        x : np.ndarray
            Bounding box in (x,y,w,h) format

        Returns
        -------
        np.ndarray
            Bounding box in (x1,y1,x2,y2) format
        """
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    @staticmethod

    @staticmethod
    def non_max_suppression_fast(boxes: np.ndarray, probs: np.ndarray, iou_threshold: float) -> List:
        """Apply non-maximum suppression to bounding boxes

        Based on:
        https://github.com/amusi/Non-Maximum-Suppression/blob/master/nms.py

        Parameters
        ----------
        boxes : np.ndarray
            Bounding boxes in (x1,y1,x2,y2) format
        probs : np.ndarray
            Confidence scores
        iou_threshold : float
            IoU threshold

        Returns
        -------
        List
            List of indexes of bounding boxes to keep
        """

        # If no bounding boxes, return empty list
        if len(boxes) == 0:
            return []

        # Bounding boxes
        boxes = np.array(boxes)

        # coordinates of bounding boxes
        start_x = boxes[:, 0]
        start_y = boxes[:, 1]
        end_x = boxes[:, 2]
        end_y = boxes[:, 3]

        # Confidence scores of bounding boxes
        score = np.array(probs)

        # Picked bounding boxes
        picked_boxes = []

        # Compute areas of bounding boxes
        areas = (end_x - start_x + 1) * (end_y - start_y + 1)

        # Sort by confidence score of bounding boxes
        order = np.argsort(score)

        # Iterate bounding boxes
        while order.size > 0:
            # The index of largest confidence score
            index = order[-1]

            # Pick the bounding box with largest confidence score
            picked_boxes.append(index)

            # Compute ordinates of intersection-over-union(IOU)
            x1 = np.maximum(start_x[index], start_x[order[:-1]])
            x2 = np.minimum(end_x[index], end_x[order[:-1]])
            y1 = np.maximum(start_y[index], start_y[order[:-1]])
            y2 = np.minimum(end_y[index], end_y[order[:-1]])

            # Compute areas of intersection-over-union
            w = np.maximum(0.0, x2 - x1 + 1)
            h = np.maximum(0.0, y2 - y1 + 1)
            intersection = w * h

            # Compute the ratio between intersection and union
            ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

            left = np.where(ratio < iou_threshold)
            order = order[left]

        return picked_boxes

    def check_loaded_model_outputs(self):
        """Check if model outputs are valid.
        Valid model are: 
            - has 1 output layer
            - output layer shape length is 3
            - batch size is 1 
        """
        if len(self.outputs_layers) == 1:
            shape = self.outputs_layers[0].shape

            if len(shape) != 3:
                raise Exception(
                    f"Detection model output should have 3 dimensions: (Batch_size, detections, values). "
                    f"Actually has: {shape}"
                )

            if shape[0] != 1:
                raise Exception(
                    f"Detection model can handle only 1-Batch outputs. Has {shape}"
                )

        else:
            raise NotImplementedError
