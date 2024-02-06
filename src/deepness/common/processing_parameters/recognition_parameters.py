import enum
from dataclasses import dataclass
from typing import Optional

from deepness.common.processing_parameters.map_processing_parameters import \
    MapProcessingParameters
from deepness.processing.models.model_base import ModelBase


@dataclass
class RecognitionParameters(MapProcessingParameters):
    """
    Parameters for Inference of Recognition model (including pre/post-processing) obtained from UI.
    """

    query_image_path: str  # path to query image
    model: ModelBase  # wrapper of the loaded model
