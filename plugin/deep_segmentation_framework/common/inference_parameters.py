from dataclasses import dataclass


@dataclass
class InferenceParameters:
    resolution_cm_per_px: float  # image resolution to used during processing
    entire_field: bool  # whether to perform operation on the entire field (otherwise on the visible map part)
    tile_size_px: int  # Tile size for processing (model input size)
    processing_overlap_percentage: float = 10.0  # aka stride - overlap of neighbouring tiles while processing

