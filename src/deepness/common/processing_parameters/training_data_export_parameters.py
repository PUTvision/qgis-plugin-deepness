from dataclasses import dataclass
from typing import Optional

from deepness.common.processing_parameters.map_processing_parameters import MapProcessingParameters


@dataclass
class TrainingDataExportParameters(MapProcessingParameters):
    """
    Parameters for Exporting Data obtained from UI.
    """

    export_image_tiles: bool  # whether to export input image tiles
    segmentation_mask_layer_id: Optional[str]  # id for mask, to be exported as separate tiles
    output_directory_path: str  # path where the output files will be saved
