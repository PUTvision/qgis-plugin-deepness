""" Module including Regression model definition
"""
from typing import List

import numpy as np

from deepness.processing.models.model_base import ModelBase


class Regressor(ModelBase):
    """ Class implements regression model.

    Regression model is used to predict metric per pixel of the image.
    """

    def __init__(self, model_file_path: str):
        """

        Parameters
        ----------
        model_file_path : str
            Path to the model file
        """
        super(Regressor, self).__init__(model_file_path)

    def postprocessing(self, model_output: List) -> np.ndarray:
        """ Postprocess the model output.

        Parameters
        ----------
        model_output : List
            Output from the (Regression) model

        Returns
        -------
        np.ndarray
            Output from the (Regression) model
        """
        return model_output

    def get_number_of_output_channels(self) -> List[int]:
        """ Returns number of channels in the output layer

        Returns
        -------
        int
            Number of channels in the output layer
        """
        channels = []

        for layer in self.outputs_layers:
            if len(layer.shape) != 4 and len(layer.shape) != 3:
                raise Exception(f'Output layer should have 3 or 4 dimensions: (Bs, H, W) or (Bs, Channels, H, W). Actually has: {layer.shape}')
            
            if len(layer.shape) == 3:
                channels.append(1)
            elif len(layer.shape) == 4:
                channels.append(layer.shape[-3])

        return channels

    @classmethod
    def get_class_display_name(cls) -> str:
        """ Returns display name of the model class

        Returns
        -------
        str
            Display name of the model class
        """
        return cls.__name__

    def check_loaded_model_outputs(self):
        """ Check if the model has correct output layers

        Correct means that:
        - there is at least one output layer
        - batch size is 1 or parameter
        - each output layer regresses only one channel
        - output resolution is square
        """
        for layer in self.outputs_layers:
            if len(layer.shape) != 4 and len(layer.shape) != 3:
                raise Exception(f'Output layer should have 3 or 4 dimensions: (Bs, H, W) or (Bs, Channels, H, W). Actually has: {layer.shape}')
            
            if len(layer.shape) == 4:
                if layer.shape[2] != layer.shape[3]:
                    raise Exception(f'Regression model can handle only square outputs masks. Has: {layer.shape}')
                
            elif len(layer.shape) == 3:
                if layer.shape[1] != layer.shape[2]:
                    raise Exception(f'Regression model can handle only square outputs masks. Has: {layer.shape}')
