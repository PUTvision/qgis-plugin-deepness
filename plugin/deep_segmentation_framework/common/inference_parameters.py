import enum
from dataclasses import dataclass

from deep_segmentation_framework.common.channels_mapping import ChannelsMapping
from deep_segmentation_framework.processing.model_wrapper import ModelWrapper


class ProcessedAreaType(enum.Enum):
    VISIBLE_PART = 'Visible part'
    ENTIRE_LAYER = 'Entire layer'
    FROM_POLYGONS = 'From polygons'

    @classmethod
    def get_all_names(cls):
        return [e.value for e in cls]


@dataclass
class InferenceParameters:
    """
    Parameters for Inference of model (including pre/post processing) obtained from UI
    """

    resolution_cm_per_px: float  # image resolution to used during processing
    processed_area_type: ProcessedAreaType  # whether to perform operation on the entire field or part
    tile_size_px: int  # Tile size for processing (model input size)
    postprocessing_dilate_erode_size: int  # dilate/erode operation size, once we have a single class map. 0 if inactive

    model: ModelWrapper  # wrapper of the loaded model

    input_layer_id: str
    mask_layer_id: str  # Processing of masked layer - if processed_area_type is FROM_POLYGONS    # TODO change to id

    input_channels_mapping: ChannelsMapping  # describes mapping of image channels to model inputs

    processing_overlap_percentage: float  # aka stride - overlap of neighbouring tiles while processing

    @property
    def tile_size_m(self):
        return self.tile_size_px * self.resolution_cm_per_px / 100

    @property
    def processing_overlap_px(self) -> int:
        """
        Always multiple of 2
        """
        return int(self.tile_size_px * self.processing_overlap_percentage / 100 * 2) // 2

    @property
    def resolution_m_per_px(self):
        return self.resolution_cm_per_px / 100

    @property
    def processing_stride_px(self):
        return self.tile_size_px - self.processing_overlap_px
