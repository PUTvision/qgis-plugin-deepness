from dataclasses import dataclass


@dataclass
class InferenceParameters:
    resolution_cm_per_px: float  # image resolution to used during processing
    entire_field: bool  # whether to perform operation on the entire field (otherwise on the visible map part)
    layer_name: str #Processing of masked layer
    tile_size_px: int  # Tile size for processing (model input size)
    postprocessing_dilate_erode_size: int  # dilate/erode operation size, once we have a single class map. 0 if inactive
    processing_overlap_percentage: float = 10.0  # aka stride - overlap of neighbouring tiles while processing

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
