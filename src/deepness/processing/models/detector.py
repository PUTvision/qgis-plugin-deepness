""" Module including the class for the object detection task and related functions
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from deepness.common.processing_parameters.detection_parameters import DetectorType
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
    mask: Optional[np.ndarray] = None
    """np.ndarray: mask of the detected object"""
    mask_offsets: Optional[Tuple[int, int]] = None

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
        
        if self.mask is not None:
            self.mask_offsets = (offset_x, offset_y)

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
        self.model_type: DetectorType | None = None
        """DetectorType: Model type"""

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

    def set_model_type_param(self, model_type: DetectorType):
        """Set model type parameters

        Parameters
        ----------
        model_type : str
            Model type
        """
        self.model_type = model_type

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
        class_names = self.get_class_names()
        if class_names is not None:
            return len(class_names)  # If class names are specified, we expect to have exactly this number of channels as specidied

        model_type_params = self.model_type.get_parameters()

        shape_index = -2 if model_type_params.has_inverted_output_shape else -1

        if len(self.outputs_layers) == 1:
            if model_type_params.skipped_objectness_probability:
                return self.outputs_layers[0].shape[shape_index] - 4
            return self.outputs_layers[0].shape[shape_index] - 4 - 1  # shape - 4 bboxes - 1 conf
        elif len(self.outputs_layers) == 2 and self.model_type == DetectorType.YOLO_ULTRALYTICS_SEGMENTATION:
            return self.outputs_layers[0].shape[shape_index] - 4 - self.outputs_layers[1].shape[1]
        else:
            raise NotImplementedError("Model with multiple output layer is not supported! Use only one output layer.")
            
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

            NOTE: Maybe refactor this, as it has many added layers of checks which can be simplified.

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

        if self.model_type is None:
            return Exception(
                "Model type is not set for model. Use self.set_model_type_param"
            )

        masks = None

        if self.model_type == DetectorType.YOLO_v5_v7_DEFAULT:
            boxes, conf, classes = self._postprocessing_YOLO_v5_v7_DEFAULT(model_output[0][0])
        elif self.model_type == DetectorType.YOLO_v6:
            boxes, conf, classes = self._postprocessing_YOLO_v6(model_output[0][0])
        elif self.model_type == DetectorType.YOLO_ULTRALYTICS:
            boxes, conf, classes = self._postprocessing_YOLO_ULTRALYTICS(model_output[0][0])
        elif self.model_type == DetectorType.YOLO_ULTRALYTICS_SEGMENTATION:
            boxes, conf, classes, masks = self._postprocessing_YOLO_ULTRALYTICS_SEGMENTATION(model_output)
        else:
            raise NotImplementedError(f"Model type not implemented! ('{self.model_type}')")

        detections = []
        
        masks = masks if masks is not None else [None] * len(boxes)

        for b, c, cl, m in zip(boxes, conf, classes, masks):
            det = Detection(
                bbox=BoundingBox(
                    x_min=b[0],
                    x_max=b[2],
                    y_min=b[1],
                    y_max=b[3]),
                conf=c,
                clss=cl,
                mask=m,
            )
            detections.append(det)

        return detections

    def _postprocessing_YOLO_v5_v7_DEFAULT(self, model_output):
        outputs_filtered = np.array(
            list(filter(lambda x: x[4] >= self.confidence, model_output))
        )

        if len(outputs_filtered.shape) < 2:
            return [], [], []

        probabilities = outputs_filtered[:, 4]

        outputs_x1y1x2y2 = self.xywh2xyxy(outputs_filtered)

        pick_indxs = self.non_max_suppression_fast(
            outputs_x1y1x2y2,
            probs=probabilities,
            iou_threshold=self.iou_threshold)

        outputs_nms = outputs_x1y1x2y2[pick_indxs]

        boxes = np.array(outputs_nms[:, :4], dtype=int)
        conf = outputs_nms[:, 4]
        classes = np.argmax(outputs_nms[:, 5:], axis=1)

        return boxes, conf, classes

    def _postprocessing_YOLO_v6(self, model_output):
        outputs_filtered = np.array(
            list(filter(lambda x: np.max(x[5:]) >= self.confidence, model_output))
        )

        if len(outputs_filtered.shape) < 2:
            return [], [], []

        probabilities = outputs_filtered[:, 4]

        outputs_x1y1x2y2 = self.xywh2xyxy(outputs_filtered)

        pick_indxs = self.non_max_suppression_fast(
            outputs_x1y1x2y2,
            probs=probabilities,
            iou_threshold=self.iou_threshold)

        outputs_nms = outputs_x1y1x2y2[pick_indxs]

        boxes = np.array(outputs_nms[:, :4], dtype=int)
        conf = np.max(outputs_nms[:, 5:], axis=1)
        classes = np.argmax(outputs_nms[:, 5:], axis=1)

        return boxes, conf, classes

    def _postprocessing_YOLO_ULTRALYTICS(self, model_output):
        model_output = np.transpose(model_output, (1, 0))

        outputs_filtered = np.array(
            list(filter(lambda x: np.max(x[4:]) >= self.confidence, model_output))
        )

        if len(outputs_filtered.shape) < 2:
            return [], [], []

        probabilities = np.max(outputs_filtered[:, 4:], axis=1)

        outputs_x1y1x2y2 = self.xywh2xyxy(outputs_filtered)

        pick_indxs = self.non_max_suppression_fast(
            outputs_x1y1x2y2,
            probs=probabilities,
            iou_threshold=self.iou_threshold)

        outputs_nms = outputs_x1y1x2y2[pick_indxs]

        boxes = np.array(outputs_nms[:, :4], dtype=int)
        conf = np.max(outputs_nms[:, 4:], axis=1)
        classes = np.argmax(outputs_nms[:, 4:], axis=1)

        return boxes, conf, classes

    def _postprocessing_YOLO_ULTRALYTICS_SEGMENTATION(self, model_output):
        detections = model_output[0][0]
        protos = model_output[1][0]
        
        detections = np.transpose(detections, (1, 0))
        
        number_of_class = self.get_number_of_output_channels()
        mask_start_index = 4 + number_of_class
        
        outputs_filtered = np.array(
            list(filter(lambda x: np.max(x[4:4+number_of_class]) >= self.confidence, detections))
        )

        if len(outputs_filtered.shape) < 2:
            return [], [], [], []
        
        probabilities = np.max(outputs_filtered[:, 4:4+number_of_class], axis=1)

        outputs_x1y1x2y2 = self.xywh2xyxy(outputs_filtered)

        pick_indxs = self.non_max_suppression_fast(
            outputs_x1y1x2y2,
            probs=probabilities,
            iou_threshold=self.iou_threshold)

        outputs_nms = outputs_x1y1x2y2[pick_indxs]

        boxes = np.array(outputs_nms[:, :4], dtype=int)
        conf = np.max(outputs_nms[:, 4:4+number_of_class], axis=1)
        classes = np.argmax(outputs_nms[:, 4:4+number_of_class], axis=1)
        masks_in = np.array(outputs_nms[:, mask_start_index:], dtype=float)
        
        masks = self.process_mask(protos, masks_in, boxes)

        return boxes, conf, classes, masks
    
    # based on https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L638C1-L638C67
    def process_mask(self, protos, masks_in, bboxes):
        c, mh, mw = protos.shape  # CHW
        ih, iw = self.input_shape[2:]

        masks = self.sigmoid(np.matmul(masks_in, protos.astype(float).reshape(c, -1))).reshape(-1, mh, mw)

        downsampled_bboxes = bboxes.copy().astype(float)
        downsampled_bboxes[:, 0] *= mw / iw
        downsampled_bboxes[:, 2] *= mw / iw
        downsampled_bboxes[:, 3] *= mh / ih
        downsampled_bboxes[:, 1] *= mh / ih

        masks = self.crop_mask(masks, downsampled_bboxes)
        scaled_masks = np.zeros((len(masks), ih, iw))

        for i in range(len(masks)):
            scaled_masks[i] = cv2.resize(masks[i], (iw, ih), interpolation=cv2.INTER_LINEAR)

        masks = np.uint8(scaled_masks >= 0.5)
            
        return masks

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    # based on https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L598C1-L614C65
    def crop_mask(masks, boxes):
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, axis=1)  # x1 shape(n,1,1)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
        c = np.arange(h, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)

        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

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
            - has 1 or 2 outputs layer
            - output layer shape length is 3
            - batch size is 1
        """

        if len(self.outputs_layers) == 1 or len(self.outputs_layers) == 2:
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
            raise NotImplementedError("Model with multiple output layer is not supported! Use only one output layer.")
