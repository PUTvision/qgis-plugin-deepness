import enum
from typing import Dict
from typing import List


class ImageChannel:
    def __init__(self, name):
        self.name = name


class ImageChannelStandaloneBand(ImageChannel):
    def __init__(self, band_no: int, name: str):
        super().__init__(name)
        self.band_no = band_no  # index within bands (counted from one)


class ImageChannelCompositeByte(ImageChannel):
    def __init__(self, byte_number: int, name: str):
        super().__init__(name)
        self.byte_number = byte_number  # position in composite byte (byte number in ARGB32, counted from zero)


class ChannelsMapping:
    """
    Defines mapping of model input channels to input image channels (bands)
    """

    def __init__(self):
        self._number_of_model_inputs = 0
        self._image_channels = []  # type: List[ImageChannel]  # what channels are available from input image

        # maps model channels to input image channels
        # model_channel_number: image_channel_index (index in self._image_channels)
        self._mapping = {}  # type: Dict[int, int]

    def set_number_of_model_inputs(self, number_of_model_inputs):
        self._number_of_model_inputs = number_of_model_inputs

    def get_number_of_model_inputs(self):
        return self._number_of_model_inputs

    def get_number_of_image_channels(self) -> int:
        return len(self._image_channels)

    def set_image_channels(self, image_channels: List[ImageChannel]):
        self._image_channels = image_channels

    def get_image_channels(self) -> List[ImageChannel]:
        return self._image_channels

    def get_image_channel_index_for_model_input(self, model_input_number) -> int:
        """
        Similar to 'get_image_channel_for_model_input', but return an index in array of inputs,
        instead of ImageChannel
        """
        image_channel_index = self._mapping.get(model_input_number, model_input_number)
        image_channel_index = min(image_channel_index, len(self._image_channels) - 1)
        return image_channel_index

    def get_image_channel_for_model_input(self, model_input_number) -> ImageChannel:
        """
        Get ImageChannel which should be used
        param: model_input_number Model input number, counted from 0
        """
        image_channel_index = self.get_image_channel_index_for_model_input(model_input_number)
        return self._image_channels[image_channel_index]

    def set_image_channel_for_model_input(self, model_input_number: int, image_channel_index: int) -> ImageChannel:
        """
        Set image_channel_index which should be used for this model input
        """
        if image_channel_index >= len(self._image_channels):
            raise Exception("Invalid image channel index!")
        image_channel = self._image_channels[image_channel_index]
        self._mapping[model_input_number] = image_channel
