"""
Raster layer (ortophoto) which is being processed consist of channels (usually Red, Green, Blue).
The neural model expects input channels with some model-defined meaning.
Channel mappings in this file define how the ortophoto channels translate to model inputs (e.g first model input is Red, second Green).
"""

import copy
from typing import Dict, List


class ImageChannel:
    """
    Defines an image channel - how is it being stored in the data source.
    See note at top of this file for details.
    """

    def __init__(self, name):
        self.name = name

    def get_band_number(self):
        raise NotImplementedError('Base class not implemented!')

    def get_byte_number(self):
        raise NotImplementedError('Base class not implemented!')


class ImageChannelStandaloneBand(ImageChannel):
    """
    Defines an image channel, where each image channel is a separate band in the data source.
    See note at top of this file for details.
    """

    def __init__(self, band_number: int, name: str):
        super().__init__(name)
        self.band_number = band_number  # index within bands (counted from one)

    def __str__(self):
        txt = f'ImageChannelStandaloneBand(name={self.name}, ' \
              f'band_number={self.band_number})'
        return txt

    def get_band_number(self):
        return self.band_number

    def get_byte_number(self):
        raise NotImplementedError('Something went wrong if we are here!')


class ImageChannelCompositeByte(ImageChannel):
    """
    Defines an image channel, where each image channel is a smaller part of a bigger value (e.g. one byte within uint32 for each pixel).
    See note at top of this file for details.
    """

    def __init__(self, byte_number: int, name: str):
        super().__init__(name)
        self.byte_number = byte_number  # position in composite byte (byte number in ARGB32, counted from zero)

    def __str__(self):
        txt = f'ImageChannelCompositeByte(name={self.name}, ' \
              f'byte_number={self.byte_number})'
        return txt

    def get_band_number(self):
        raise NotImplementedError('Something went wrong if we are here!')

    def get_byte_number(self):
        return self.byte_number


class ChannelsMapping:
    """
    Defines mapping of model input channels to input image channels (bands).
    See note at top of this file for details.
    """

    INVALID_INPUT_CHANNEL = -1

    def __init__(self):
        self._number_of_model_inputs = 0
        self._number_of_model_output_channels = 0
        self._image_channels = []  # type: List[ImageChannel]  # what channels are available from input image

        # maps model channels to input image channels
        # model_channel_number: image_channel_index (index in self._image_channels)
        self._mapping = {}  # type: Dict[int, int]

    def __str__(self):
        txt = f'ChannelsMapping(' \
              f'number_of_model_inputs={self._number_of_model_inputs}, ' \
              f'image_channels = {self._image_channels}, ' \
              f'mapping {self._mapping})'
        return txt

    def __eq__(self, other):
        if self._number_of_model_inputs != other._number_of_model_inputs:
            return False
        return True

    def get_as_default_mapping(self):
        """
        Get the same channels mapping as we have right now, but without the mapping itself
        (so just a definition of inputs and outputs)

        Returns
        -------
        ChannelsMapping
        """
        default_channels_mapping = copy.deepcopy(self)
        default_channels_mapping._mapping = {}
        return default_channels_mapping

    def are_all_inputs_standalone_bands(self):
        """
        Checks whether all image_channels are standalone bands (ImageChannelStandaloneBand)
        """
        for image_channel in self._image_channels:
            if not isinstance(image_channel, ImageChannelStandaloneBand):
                return False
        return True

    def are_all_inputs_composite_byte(self):
        """
        Checks whether all image_channels are composite byte (ImageChannelCompositeByte)
        """
        for image_channel in self._image_channels:
            if not isinstance(image_channel, ImageChannelCompositeByte):
                return False
        return True

    def set_number_of_model_inputs(self, number_of_model_inputs: int):
        """ Set how many input channels does the model has
        Parameters
        ----------
        number_of_model_inputs : int
        """
        self._number_of_model_inputs = number_of_model_inputs

    def set_number_of_model_output_channels(self, number_of_output_channels: int):
        """ Set how many output channels does the model has

        Parameters
        ----------
        number_of_output_channels : int
        """
        self._number_of_model_output_channels = number_of_output_channels

    def set_number_of_model_inputs_same_as_image_channels(self):
        """ Set the number of model input channels to be the same as number of image channels
        """
        self._number_of_model_inputs = len(self._image_channels)

    def get_number_of_model_inputs(self) -> int:
        """ Get number of model input channels

        Returns
        -------
        int
        """
        return self._number_of_model_inputs

    def get_number_of_model_output_channels(self) -> int:
        """ Get number of model output channels

        Returns
        -------
        int
        """
        return self._number_of_model_output_channels

    def get_number_of_image_channels(self) -> int:
        """ Get number of image input channels

        Returns
        -------
        int
        """
        return len(self._image_channels)

    def set_image_channels(self, image_channels: List[ImageChannel]):
        """ Set what are the image channels

        Parameters
        ----------
        image_channels : List[ImageChannel]
            Image channels to set
        """
        self._image_channels = image_channels
        if not self.are_all_inputs_standalone_bands() and not self.are_all_inputs_composite_byte():
            raise Exception("Unsupported image channels composition!")

    def get_image_channels(self) -> List[ImageChannel]:
        """ Get the current image channels definition

        Returns
        -------
        List[ImageChannel]
        """
        return self._image_channels

    def get_image_channel_index_for_model_input(self, model_input_number) -> int:
        """
        Similar to 'get_image_channel_for_model_input', but return an index in array of inputs,
        instead of ImageChannel
        """
        if len(self._image_channels) == 0:
            raise Exception("No image channels!")

        image_channel_index = self._mapping.get(model_input_number, model_input_number)
        image_channel_index = min(image_channel_index, len(self._image_channels) - 1)
        return image_channel_index

    def get_image_channel_for_model_input(self, model_input_number: int) -> ImageChannel:
        """
        Get ImageChannel which should be used for the specified model input

        Parameters
        ----------
        model_input_number : int
            Model input number, counted from 0

        Returns
        -------
        ImageChannel
        """
        image_channel_index = self.get_image_channel_index_for_model_input(model_input_number)
        return self._image_channels[image_channel_index]

    def set_image_channel_for_model_input(self, model_input_number: int, image_channel_index: int) -> ImageChannel:
        """
        Set image_channel_index which should be used for this model input
        """
        if image_channel_index >= len(self._image_channels):
            raise Exception("Invalid image channel index!")
        # image_channel = self._image_channels[image_channel_index]
        self._mapping[model_input_number] = image_channel_index

    def get_mapping_as_list(self) -> List[int]:
        """ Get the mapping of model input channels to image channels, but as a list (e.g. to store it in QGis configuration)

        Returns
        -------
        List[int]
        """
        mapping_list = []
        for i in range(self._number_of_model_inputs):
            if i in self._mapping:
                mapping_list.append(self._mapping[i])
            else:
                mapping_list.append(-1)
        return mapping_list

    def load_mapping_from_list(self, mapping_list: List[int]):
        """
        Load self._mapping from a plain list of channels (which is saved in config)
        """
        for i in range(min(self._number_of_model_inputs), len(mapping_list)):
            proposed_channel = mapping_list[i]
            if proposed_channel == -1 or proposed_channel >= self._number_of_model_inputs:
                continue

            self._mapping[i] = proposed_channel
