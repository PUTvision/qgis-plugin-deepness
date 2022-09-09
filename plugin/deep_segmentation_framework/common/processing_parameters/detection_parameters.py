import enum
from dataclasses import dataclass
from typing import Optional

from deep_segmentation_framework.common.channels_mapping import ChannelsMapping
from deep_segmentation_framework.common.processing_parameters.map_processing_parameters import MapProcessingParameters
from deep_segmentation_framework.processing.model_wrapper import ModelWrapper
from deep_segmentation_framework.processing.models.detector import Detector


@dataclass
class DetectionParameters(MapProcessingParameters):
    """
    Parameters for Inference of detection model (including pre/post-processing) obtained from UI.
    """

    model: Detector  # wrapper of the loaded model

    score_threshold: float
    iou_threshold: float
    