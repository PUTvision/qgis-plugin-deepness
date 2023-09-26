import enum
from typing import Dict, List


class ProcessingOverlapOptions(enum.Enum):
    OVERLAP_IN_PIXELS = 'Overlap in pixels'
    OVERLAP_IN_PERCENT = 'Overlap in percent'


class ProcessingOverlap:
    """ Represents overlap between tiles during processing
    """
    def __init__(self, selected_option: ProcessingOverlapOptions, percentage: float = None, overlap_px: int = None):
        self.selected_option = selected_option
        
        if selected_option == ProcessingOverlapOptions.OVERLAP_IN_PERCENT and percentage is None:
            raise Exception(f"Percentage must be specified when using {ProcessingOverlapOptions.OVERLAP_IN_PERCENT}")
        if selected_option == ProcessingOverlapOptions.OVERLAP_IN_PIXELS and overlap_px is None:
            raise Exception(f"Overlap in pixels must be specified when using {ProcessingOverlapOptions.OVERLAP_IN_PIXELS}")

        if selected_option == ProcessingOverlapOptions.OVERLAP_IN_PERCENT:
            self._percentage = percentage
        elif selected_option == ProcessingOverlapOptions.OVERLAP_IN_PIXELS:
            self._overlap_px = overlap_px
        else:
            raise Exception(f"Unknown option: {selected_option}")

    def get_overlap_px(self, tile_size_px: int) -> int:
        """ Returns the overlap in pixels

        :param tile_size_px: Tile size in pixels
        :return: Returns the overlap in pixels
        """
        if self.selected_option == ProcessingOverlapOptions.OVERLAP_IN_PIXELS:
            return self._overlap_px
        else:
            return int(tile_size_px * self._percentage / 100 * 2) // 2  # TODO: check if this is correct
