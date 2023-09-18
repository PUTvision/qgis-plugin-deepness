""" Module including Super Resolution model definition
"""
from typing import List
import numpy as np

from deepness.processing.models.model_base import ModelBase


class Superresolution(ModelBase):
    """ Class implements super resolution model.

    Super Resolution  model is used improve the resolution of an image.
    """

    def __init__(self, model_file_path: str):
        """

        Parameters
        ----------
        model_file_path : str
            Path to the model file
        """
        super(Superresolution, self).__init__(model_file_path)

    def preprocessing(self, image: np.ndarray) -> np.ndarray:
        """ Preprocess the image for the model (resize, normalization, etc)

        Parameters
        ----------
        image : np.ndarray
            Image to preprocess (H,W,C), RGB, 0-255

        Returns
        -------
        np.ndarray
            Preprocessed image (1,C,H,W), RGB, 0-1
        """
        img = image[:, :, :self.input_shape[-3]]

        input_batch = img.astype('float32')
        input_batch /= 255
        input_batch = input_batch.transpose(2, 0, 1)
        input_batch = np.expand_dims(input_batch, axis=0)

        return input_batch

    def postprocessing(self, model_output: List) -> np.ndarray:
        """ Postprocess the model output.

        Parameters
        ----------
        model_output : List
            Output from the (Regression) model

        Returns
        -------
        np.ndarray
            Postprocessed mask (H,W,C), 0-1 (one output channel)

        """
        return model_output[0][0]

    def get_number_of_output_channels(self) -> int:
        """ Returns number of channels in the output layer

        Returns
        -------
        int
            Number of channels in the output layer
        """
        if len(self.outputs_layers) == 1:
            return self.outputs_layers[0].shape[-3]
        else:
            raise NotImplementedError("Model with multiple output layers is not supported! Use only one output layer.")

    @classmethod
    def get_class_display_name(cls) -> str:
        """ Returns display name of the model class

        Returns
        -------
        str
            Display name of the model class
        """
        return cls.__name__

    def get_output_shape(self) -> List[int]:
        """ Returns shape of the output layer

        Returns
        -------
        List[int]
            Shape of the output layer
        """
        if len(self.outputs_layers) == 1:
            return self.outputs_layers[0].shape
        else:
            raise NotImplementedError("Model with multiple output layers is not supported! Use only one output layer.")

    def check_loaded_model_outputs(self):
        """ Check if the model has correct output layers

        Correct means that:
        - there is only one output layer
        - output layer has 1 channel
        - batch size is 1
        - output resolution is square
        """
        if len(self.outputs_layers) == 1:
            shape = self.outputs_layers[0].shape

            if len(shape) != 4:
                raise Exception(f'Regression model output should have 4 dimensions: (Batch_size, Channels, H, W). \n'
                                f'Actually has: {shape}')

            if shape[0] != 1:
                raise Exception(f'Regression model can handle only 1-Batch outputs. Has {shape}')

            if shape[2] != shape[3]:
                raise Exception(f'Regression model can handle only square outputs masks. Has: {shape}')

        else:
            raise NotImplementedError("Model with multiple output layers is not supported! Use only one output layer.")
