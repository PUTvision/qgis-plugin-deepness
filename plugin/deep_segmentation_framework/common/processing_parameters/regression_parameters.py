import enum
from dataclasses import dataclass
from typing import Optional

from deep_segmentation_framework.common.channels_mapping import ChannelsMapping
from deep_segmentation_framework.common.processing_parameters.map_processing_parameters import MapProcessingParameters
from deep_segmentation_framework.processing.models.model_base import ModelBase


@dataclass
class RegressionParameters(MapProcessingParameters):
    """
    Parameters for Inference of Regression model (including pre/post-processing) obtained from UI.
    """

    output_scaling: float  # scaling factor for the model output (keep 1 if maximum model output value is 1)
    model: ModelBase  # wrapper of the loaded model
