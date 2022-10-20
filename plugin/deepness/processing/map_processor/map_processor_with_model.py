
""" This file implements map processing functions common for all map processors using nural model """

from typing import List

from deepness.common.processing_parameters.map_processing_parameters import ModelOutputFormat
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

        output_channels = []
        if self.params.model_output_format == ModelOutputFormat.ONLY_SINGLE_CLASS_AS_LAYER:
            channel = self.params.model_output_format__single_class_number
            if channel >= self.model.get_number_of_output_channels():
                # we shouldn't get here, it should not be allowed to select it in the UI
                raise Exception("Cannot get a bigger output channel than number of model outputs!")
            output_channels.append(channel)
        elif self.params.model_output_format == ModelOutputFormat.ALL_CLASSES_AS_SEPARATE_LAYERS:
            output_channels = list(range(0, self.model.get_number_of_output_channels()))
        elif self.params.model_output_format == ModelOutputFormat.CLASSES_AS_SEPARATE_LAYERS_WITHOUT_ZERO_CLASS:
            output_channels = list(range(1, self.model.get_number_of_output_channels()))
        else:
            raise Exception(f"Unhandled model output format {self.params.model_output_format}")

        return output_channels
