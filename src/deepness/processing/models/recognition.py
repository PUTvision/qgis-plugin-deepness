""" Module including the class for the recognition of the images
"""
import logging
from typing import List

import numpy as np
from deepness.common.lazy_package_loader import LazyPackageLoader
from deepness.processing.models.model_base import ModelBase

cv2 = LazyPackageLoader('cv2')
    
IMG_SIZE = 224
mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)

def normalize(img, mean, std, max_pixel_value=255.0):
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    
    return img

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
        
        
    def process(self, img):
        """Process a single tile image

        Parameters
        ----------
        img : np.ndarray
            Image to process ([TILE_SIZE x TILE_SIZE x channels], type uint8, values 0 to 255)

        Returns
        -------
        np.ndarray
            embeddings
        """
        input_batch = self.preprocessing(img)
        model_output = self.sess.run(
            output_names=None, input_feed={self.input_name: input_batch}
        )
        res = self.postprocessing(model_output)
        return res


    def preprocessing(self, image: np.ndarray):
        """Preprocess image before inference

        Parameters
        ----------
        image : np.ndarray
            Image to preprocess in RGB format

        Returns
        -------
        np.ndarray
            Preprocessed image
        """
        img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        img = img[:, :, : self.input_shape[-3]]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_batch = img.astype("float32")
        input_batch = normalize(input_batch, mean, std, max_pixel_value=255.0)
        input_batch = input_batch.transpose(2, 0, 1)
        input_batch = np.expand_dims(input_batch, axis=0)

        return input_batch

    def postprocessing(self, model_output: List) -> np.ndarray:
        """Postprocess the model output.
        Function returns the array of embeddings
        
        Anything to do here?

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
            #cannot, won't work for query image

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
        print(f"outputs_layers: {self.outputs_layers}")

        if len(self.outputs_layers) == 1:
            return self.outputs_layers[0].shape[1]
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

            if shape[0] != 1:
                raise Exception(
                    f"Recognition model can handle only 1-Batch outputs. Has {shape}"
                )

        else:
            raise NotImplementedError(
                "Model with multiple output layers is not supported! Use only one output layer."
            )
