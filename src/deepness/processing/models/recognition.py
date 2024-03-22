""" Module including the class for the recognition of the images
"""
import logging
from typing import List

import numpy as np

from deepness.common.lazy_package_loader import LazyPackageLoader
from deepness.processing.models.model_base import ModelBase

cv2 = LazyPackageLoader('cv2')


class Recognition(ModelBase):
    """Class implements recognition model

    Recognition model is used to predict class confidence per pixel of the image.
    """

    def __init__(self, model_file_path: str):
        """

        Parameters
        ----------
        model_file_path : str
            Path to the model file
        """
        super(Recognition, self).__init__(model_file_path)

    def postprocessing(self, model_output: List) -> np.ndarray:
        """Postprocess the model output.
        Function returns the array of embeddings
        
        Parameters
        ----------
        model_output : List
            Output embeddings from the (Recognition) model

        Returns
        -------
        np.ndarray
            Same as input
        """
        # TODO - compute cosine similarity to self.query_img_emb
        # cannot, won't work for query image

        return np.array(model_output)

    def get_number_of_output_channels(self):
        """Returns model's number of class

        Returns
        -------
        int
            Number of channels in the output layer
        """
        logging.warning(f"outputs_layers: {self.outputs_layers}")
        logging.info(f"outputs_layers: {self.outputs_layers}")

        if len(self.outputs_layers) == 1:
            return [self.outputs_layers[0].shape[1]]
        else:
            raise NotImplementedError(
                "Model with multiple output layers is not supported! Use only one output layer."
            )

    @classmethod
    def get_class_display_name(cls):
        """Returns the name of the class to be displayed in the GUI

        Returns
        -------
        str
            Name of the class
        """
        return cls.__name__

    def check_loaded_model_outputs(self):
        """Checks if the model outputs are valid

        Valid means that:
        - the model has only one output
        - the output is 2D (N,C)
        - the batch size is 1

        """
        if len(self.outputs_layers) == 1:
            shape = self.outputs_layers[0].shape

            if len(shape) != 2:
                raise Exception(
                    f"Recognition model output should have 4 dimensions: (B,C,H,W). Has {shape}"
                )

        else:
            raise NotImplementedError(
                "Model with multiple output layers is not supported! Use only one output layer."
            )
