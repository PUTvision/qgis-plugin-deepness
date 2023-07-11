from dataclasses import dataclass

from deepness.common.processing_parameters.map_processing_parameters import MapProcessingParameters
from deepness.processing.models.model_base import ModelBase


@dataclass
class DetectionParameters(MapProcessingParameters):
    """
    Parameters for Inference of detection model (including pre/post-processing) obtained from UI.
    """

    model: ModelBase  # wrapper of the loaded model

    confidence: float
    iou_threshold: float
    remove_overlapping_detections: bool  # whether overlapping detections can be deleted
