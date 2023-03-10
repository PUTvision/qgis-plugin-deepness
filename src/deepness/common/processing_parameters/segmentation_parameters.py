import enum
from dataclasses import dataclass
from typing import Optional

from deepness.common.channels_mapping import ChannelsMapping
from deepness.common.processing_parameters.map_processing_parameters import MapProcessingParameters
from deepness.processing.models.model_base import ModelBase


@dataclass
class SegmentationParameters(MapProcessingParameters):
    """
    Parameters for Inference of Segmentation model (including pre/post-processing) obtained from UI.
    """

    postprocessing_dilate_erode_size: int  # dilate/erode operation size, once we have a single class map. 0 if inactive. Implementation may use median filer instead of erode/dilate
    model: ModelBase  # wrapper of the loaded model

    pixel_classification__probability_threshold: float  # Minimum required class probability for pixel. 0 if disabled
