import enum
from dataclasses import dataclass
from typing import Optional

from deep_segmentation_framework.common.channels_mapping import ChannelsMapping
from deep_segmentation_framework.common.processing_parameters.map_processing_parameters import MapProcessingParameters
from deep_segmentation_framework.processing.models.model_base import ModelBase
from deep_segmentation_framework.processing.models.detector import Detector


@dataclass
class DetectionParameters(MapProcessingParameters):
    """
    Parameters for Inference of detection model (including pre/post-processing) obtained from UI.
    """

    model: ModelBase  # wrapper of the loaded model

    confidence: float
    iou_threshold: float
    