""" Module including the class for the segmentation of the images
"""
from typing import List

import numpy as np

from deepness.processing.models.model_base import ModelBase


class Segmentor(ModelBase):
    """Class implements segmentation model

    Segmentation model is used to predict class confidence per pixel of the image.
    """

    def __init__(self, model_file_path: str):
        """

        Parameters
        ----------
        model_file_path : str
            Path to the model file
        """
        super(Segmentor, self).__init__(model_file_path)

    def preprocessing(self, image: np.ndarray):
        """ Preprocess the image for the model

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
        Function returns the mask with the probability of the presence of the class in the image.

        Parameters
        ----------
        model_output : List
            Output from the (Segmentation) model

        Returns
        -------
        np.ndarray
            Postprocessed mask (H,W,C), 0-1
        """
        labels = np.clip(model_output[0][0], 0, 1)

        return labels

    def get_number_of_output_channels(self):
        """ Returns model's number of class

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
    def get_class_display_name(cls):
        """ Returns the name of the class to be displayed in the GUI

        Returns
        -------
        str
            Name of the class
        """
        return cls.__name__

    def check_loaded_model_outputs(self):
        """ Checks if the model outputs are valid

        Valid means that:
        - the model has only one output
        - the output is 4D (N,C,H,W)
        - the batch size is 1
        - model resolution is equal to TILE_SIZE (is square)

        """
        if len(self.outputs_layers) == 1:
            shape = self.outputs_layers[0].shape

            if len(shape) != 4:
                raise Exception(f'Segmentation model output should have 4 dimensions: (B,C,H,W). Has {shape}')

            if shape[0] != 1:
                raise Exception(f'Segmentation model can handle only 1-Batch outputs. Has {shape}')

            if shape[2] != shape[3]:
                raise Exception(f'Segmentation model can handle only square outputs masks. Has: {shape}')

        else:
            raise NotImplementedError("Model with multiple output layers is not supported! Use only one output layer.")
