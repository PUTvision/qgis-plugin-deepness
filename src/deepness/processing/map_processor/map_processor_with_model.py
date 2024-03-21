
""" This file implements map processing functions common for all map processors using nural model """

from typing import List

from deepness.processing.map_processor.map_processor import MapProcessor
from deepness.processing.models.model_base import ModelBase


class MapProcessorWithModel(MapProcessor):
    """
    Common base class for MapProcessor with models
    """

    def __init__(self,
                 model: ModelBase,
                 **kwargs):
        super().__init__(
            **kwargs)
        self.model = model

    def _get_indexes_of_model_output_channels_to_create(self) -> List[int]:
        """
        Decide what model output channels/classes we want to use at presentation level
        (e.g. for which channels create a layer with results)
        """
        return self.model.get_number_of_output_channels()
