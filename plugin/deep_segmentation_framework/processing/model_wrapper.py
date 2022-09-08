import numpy as np

from deep_segmentation_framework.processing.models.segmenter import Segmentor


class ModelWrapper:
    """
    Wraps the ONNX model used during processing into a common interface
    """

    def __init__(self, model_file_path):
        self.model_file_path = model_file_path
        self.model = Segmentor(self.model_file_path)

    def get_input_shape(self):
        """
        Get shape of the input for the model
        """
        return self.model.input_shape

    def get_input_size_in_pixels(self):
        """
        Get number of input pixels in x and y direction (the same value)
        """
        return self.model.input_shape[-2:]

    def get_number_of_channels(self):
        return self.model.input_shape[-3]

    def process(self, img):
        """
        Process a single tile image
        :param img: RGB img [TILE_SIZE x TILE_SIZE x channels], type uint8, values 0 to 255
        :return: single prediction
        """

        model_output = self.model.predict(img)

        return model_output
