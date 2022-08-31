import enum
from typing import Dict
from typing import List


class ImageChannel:
    pass


class ImageChannelStandaloneBand(ImageChannel):
    def __init__(self, band_no: int, name: str):
        self.band_no = band_no  # index within bands (counted from one)
        self.name = name


class ImageChannelCompositeByte(ImageChannel):
    def __init__(self, byte_number: int, name: str):
        self.byte_number = byte_number  # position in composite byte (byte number in ARGB32, counted from zero)
        self.name = name


class ChannelsMapping:
    def __init__(self):
        self._number_of_model_inputs = 0
        self._image_channels = []  # type: List[ImageChannel]  # what channels are available from input image

        # maps model channels to input channels
        # model_channel_number: image_channel_index (index in self._image_channels)
        self._mapping = {}  # type: Dict[int, int]

    def set_number_of_model_inputs(self, number_of_model_inputs):
        self._number_of_model_inputs = number_of_model_inputs

    def set_image_channels(self, image_channels: List[ImageChannel]):
        self._image_channels = image_channels

    def get_image_channel_for_model_input(self, model_input_number) -> ImageChannel:
        """
        Get ImageChannel which should be used
        param: model_input_number Model input number, counted from 0
        """
        self._mapping.get(model_input_number, model_input_number)