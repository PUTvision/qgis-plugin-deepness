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
            Output from the (Segmentation) model
        """
        return model_output

    def get_number_of_output_channels(self) -> List[int]:
        """ Returns model's number of class

        Returns
        -------
        int
            Number of channels in the output layer
        """
        output_channels = []
        for layer in self.outputs_layers:
            ls = layer.shape

            if len(ls) == 3:
                output_channels.append(1)
            elif len(ls) == 4:
                output_channels.append(ls[-3])

        return output_channels

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
        - the model has at least one output
        - the output is 4D (N,C,H,W) or 3D (N,H,W)
        - the batch size is 1 or dynamic
        - model resolution is equal to TILE_SIZE (is square)

        """
        if len(self.outputs_layers) == 0:
            raise Exception('Model has no output layers')

        for layer in self.outputs_layers:
            if len(layer.shape) != 4 and len(layer.shape) != 3:
                raise Exception(f'Segmentation model output should have 4 dimensions: (B,C,H,W) or 3 dimensions: (B,H,W). Has {layer.shape}')
