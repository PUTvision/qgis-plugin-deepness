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

        self.outputs_are_sigmoid = self.check_loaded_model_outputs()

        for idx in range(len(self.outputs_layers)):
            if self.outputs_names is None:
                continue

            if len(self.outputs_names[idx]) == 1 and self.outputs_are_sigmoid[idx]:
                self.outputs_names[idx] = ['background', self.outputs_names[idx][0]]

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
                output_channels.append(2)
            elif len(ls) == 4:
                chn = ls[-3]
                if chn == 1:
                    output_channels.append(2)
                else:
                    output_channels.append(chn)

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

    def check_loaded_model_outputs(self) -> List[bool]:
        """ Check if the model outputs are sigmoid (for segmentation)

        Parameters
        ----------

        Returns
        -------
        List[bool]
            List of booleans indicating if the model outputs are sigmoid
        """
        outputs = []

        for output in self.outputs_layers:
            if len(output.shape) == 3:
                outputs.append(True)
            else:
                outputs.append(output.shape[-3] == 1)

        return outputs
