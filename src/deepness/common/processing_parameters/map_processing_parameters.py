import enum
from dataclasses import dataclass
from typing import Optional

from deepness.common.channels_mapping import ChannelsMapping
from deepness.common.processing_overlap import ProcessingOverlap


class ProcessedAreaType(enum.Enum):
    VISIBLE_PART = 'Visible part'
    ENTIRE_LAYER = 'Entire layer'
    FROM_POLYGONS = 'From polygons'

    @classmethod
    def get_all_names(cls):
        return [e.value for e in cls]


class ModelOutputFormat(enum.Enum):
    ALL_CLASSES_AS_SEPARATE_LAYERS = 'All classes as separate layers'
    CLASSES_AS_SEPARATE_LAYERS_WITHOUT_ZERO_CLASS = 'Classes as separate layers (without 0 class)'
    ONLY_SINGLE_CLASS_AS_LAYER = 'Single class as a vector layer'

    @classmethod
    def get_all_names(cls):
        return [e.value for e in cls]


@dataclass
class MapProcessingParameters:
    """
    Common parameters for map processing obtained from UI.

    TODO: Add default values here, to later set them in UI at startup
    """

    resolution_cm_per_px: float  # image resolution to used during processing
    processed_area_type: ProcessedAreaType  # whether to perform operation on the entire field or part
    tile_size_px: int  # Tile size for processing (model input size)
    batch_size: int  # Batch size for processing
    local_cache: bool  # Whether to use local cache for tiles (on disk, /tmp directory)

    input_layer_id: str  # raster layer to process
    mask_layer_id: Optional[str]  # Processing of masked layer - if processed_area_type is FROM_POLYGONS

    processing_overlap: ProcessingOverlap  # aka "stride" - how much to overlap tiles during processing

    input_channels_mapping: ChannelsMapping  # describes mapping of image channels to model inputs

    model_output_format: ModelOutputFormat  # what kind of model output do we want to achieve
    model_output_format__single_class_number: int  # if we want to show just one output channel - here is its number

    @property
    def tile_size_m(self):
        return self.tile_size_px * self.resolution_cm_per_px / 100

    @property
    def processing_overlap_px(self) -> int:
        """
        Always divisible by 2, because overlap is on both sides of the tile
        """
        return self.processing_overlap.get_overlap_px(self.tile_size_px)

    @property
    def resolution_m_per_px(self):
        return self.resolution_cm_per_px / 100

    @property
    def processing_stride_px(self):
        return self.tile_size_px - self.processing_overlap_px
