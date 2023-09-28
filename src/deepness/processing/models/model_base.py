""" Module including the base model interfaces and utilities"""
import ast
import json
from typing import List, Optional

import numpy as np

from deepness.common.lazy_package_loader import LazyPackageLoader

ort = LazyPackageLoader('onnxruntime')


class ModelBase:
    """
    Wraps the ONNX model used during processing into a common interface
    """

    def __init__(self, model_file_path: str):
        """

        Parameters
        ----------
        model_file_path : str
            Path to the model file
        """
        self.model_file_path = model_file_path

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = [
            'CUDAExecutionProvider',
            'CPUExecutionProvider'
        ]

        self.sess = ort.InferenceSession(self.model_file_path, options=options, providers=providers)
        inputs = self.sess.get_inputs()
        if len(inputs) > 1:
            raise Exception("ONNX model: unsupported number of inputs")
        input_0 = inputs[0]

        self.input_shape = input_0.shape
        self.input_name = input_0.name

        self.outputs_layers = self.sess.get_outputs()

    @classmethod
    def get_model_type_from_metadata(cls, model_file_path: str) -> Optional[str]:
        """ Get model type from metadata

        Parameters
        ----------
        model_file_path : str
            Path to the model file

        Returns
        -------
        Optional[str]
            Model type or None if not found
        """
        model = cls(model_file_path)
        return model.get_metadata_model_type()

    def get_input_shape(self) -> tuple:
        """ Get shape of the input for the model

        Returns
        -------
        tuple
            Shape of the input (batch_size, channels, height, width)
        """
        return self.input_shape

    def get_input_size_in_pixels(self) -> int:
        """ Get number of input pixels in x and y direction (the same value)

        Returns
        -------
        int
            Number of pixels in x and y direction
        """
        return self.input_shape[-2:]

    def get_class_names(self) -> Optional[List[str]]:
        """ Get class names from metadata

        Returns
        -------
        List[str] | None
            List of class names or None if not found
        """
        meta = self.sess.get_modelmeta()

        allowed_key_names = ['class_names', 'names']  # support both names for backward compatibility
        for name in allowed_key_names:
            if name not in meta.custom_metadata_map:
                continue

            txt = meta.custom_metadata_map[name]
            try:
                class_names = json.loads(txt)  # default format recommended in the documentation - classes encoded as json
            except json.decoder.JSONDecodeError:
                class_names = ast.literal_eval(txt)  # keys are integers instead of strings - use ast

            sorted_by_key = sorted(class_names.items(), key=lambda kv: int(kv[0]))

            class_counter = 0
            all_names = []
            for key, value in sorted_by_key:
                if int(key) != class_counter:
                    raise Exception("Class names in the model metadata are not consecutive (missing class label)")
                class_counter += 1
                all_names.append(value)

            return all_names

        return None

    def get_channel_name(self, channel_id: int) -> str:
        """ Get channel name by id if exists in model metadata

        Parameters
        ----------
        channel_id : int
            Channel id (means index in the output tensor)

        Returns
        -------
        str
            Channel name or empty string if not found
        """
        class_names = self.get_class_names()
        channel_id_str = str(channel_id)
        default_return = f'channel_{channel_id_str}'

        if class_names is not None and channel_id < len(class_names):
            return class_names[channel_id]
        else:
            return default_return

    def get_metadata_model_type(self) -> Optional[str]:
        """ Get model type from metadata

        Returns
        -------
        Optional[str]
            Model type or None if not found
        """
        meta = self.sess.get_modelmeta()
        name = 'model_type'
        if name in meta.custom_metadata_map:
            value = json.loads(meta.custom_metadata_map[name])
            return str(value).capitalize()
        return None

    def get_metadata_resolution(self) -> Optional[float]:
        """ Get resolution from metadata if exists

        Returns
        -------
        Optional[float]
            Resolution or None if not found
        """
        meta = self.sess.get_modelmeta()
        name = 'resolution'
        if name in meta.custom_metadata_map:
            value = json.loads(meta.custom_metadata_map[name])
            return float(value)
        return None

    def get_metadata_tile_size(self) -> Optional[int]:
        """ Get tile size from metadata if exists

        Returns
        -------
        Optional[int]
            Tile size or None if not found
        """
        meta = self.sess.get_modelmeta()
        name = 'tile_size'
        if name in meta.custom_metadata_map:
            value = json.loads(meta.custom_metadata_map[name])
            return int(value)
        return None

    def get_metadata_tiles_overlap(self) -> Optional[int]:
        """ Get tiles overlap from metadata if exists

        Returns
        -------
        Optional[int]
            Tiles overlap or None if not found
        """
        meta = self.sess.get_modelmeta()
        name = 'tiles_overlap'
        if name in meta.custom_metadata_map:
            value = json.loads(meta.custom_metadata_map[name])
            return int(value)
        return None

    def get_metadata_segmentation_threshold(self) -> Optional[float]:
        """ Get segmentation threshold from metadata if exists

        Returns
        -------
        Optional[float]
            Segmentation threshold or None if not found
        """
        meta = self.sess.get_modelmeta()
        name = 'seg_thresh'
        if name in meta.custom_metadata_map:
            value = json.loads(meta.custom_metadata_map[name])
            return float(value)
        return None

    def get_metadata_segmentation_small_segment(self) -> Optional[int]:
        """ Get segmentation small segment from metadata if exists

        Returns
        -------
        Optional[int]
            Segmentation small segment or None if not found
        """
        meta = self.sess.get_modelmeta()
        name = 'seg_small_segment'
        if name in meta.custom_metadata_map:
            value = json.loads(meta.custom_metadata_map[name])
            return int(value)
        return None

    def get_metadata_regression_output_scaling(self) -> Optional[float]:
        """ Get regression output scaling from metadata if exists

        Returns
        -------
        Optional[float]
            Regression output scaling or None if not found
        """
        meta = self.sess.get_modelmeta()
        name = 'reg_output_scaling'
        if name in meta.custom_metadata_map:
            value = json.loads(meta.custom_metadata_map[name])
            return float(value)
        return None

    def get_metadata_detection_confidence(self) -> Optional[float]:
        """ Get detection confidence from metadata if exists

        Returns
        -------
        Optional[float]
            Detection confidence or None if not found
        """
        meta = self.sess.get_modelmeta()
        name = 'det_conf'
        if name in meta.custom_metadata_map:
            value = json.loads(meta.custom_metadata_map[name])
            return float(value)
        return None

    def get_detector_type(self) -> Optional[str]:
        """ Get detector type from metadata if exists

        Returns string value of DetectorType enum or None if not found
        -------
        Optional[str]
            Detector type or None if not found
        """
        meta = self.sess.get_modelmeta()
        name = 'det_type'
        if name in meta.custom_metadata_map:
            value = json.loads(meta.custom_metadata_map[name])
            return str(value)
        return None

    def get_metadata_detection_iou_threshold(self) -> Optional[float]:
        """ Get detection iou threshold from metadata if exists

        Returns
        -------
        Optional[float]
            Detection iou threshold or None if not found
        """
        meta = self.sess.get_modelmeta()
        name = 'det_iou_thresh'
        if name in meta.custom_metadata_map:
            value = json.loads(meta.custom_metadata_map[name])
            return float(value)
        return None

    def get_metadata_detection_remove_overlapping(self) -> Optional[bool]:
        """ Get detection parameter 'should remove overlapping detections' from metadata if exists
        """
        meta = self.sess.get_modelmeta()
        name = 'det_remove_overlap'
        if name in meta.custom_metadata_map:
            value = json.loads(meta.custom_metadata_map[name])
            return bool(value)
        return None

    def get_number_of_channels(self) -> int:
        """ Returns number of channels in the input layer

        Returns
        -------
        int
            Number of channels in the input layer
        """
        return self.input_shape[-3]

    def process(self, img):
        """ Process a single tile image

        Parameters
        ----------
        img : np.ndarray
            Image to process ([TILE_SIZE x TILE_SIZE x channels], type uint8, values 0 to 255)

        Returns
        -------
        np.ndarray
            Single prediction
        """
        input_batch = self.preprocessing(img)
        model_output = self.sess.run(
            output_names=None,
            input_feed={self.input_name: input_batch})
        res = self.postprocessing(model_output)
        return res

    def preprocessing(self, img: np.ndarray) -> np.ndarray:
        """ Abstract method for preprocessing

        Parameters
        ----------
        img : np.ndarray
            Image to process ([TILE_SIZE x TILE_SIZE x channels], type uint8, values 0 to 255, RGB order)

        Returns
        -------
        np.ndarray
            Preprocessed image
        """
        raise NotImplementedError('Base class not implemented!')

    def postprocessing(self, outs: List) -> np.ndarray:
        """ Abstract method for postprocessing

        Parameters
        ----------
        outs : List
            Output from the model (depends on the model type)

        Returns
        -------
        np.ndarray
            Postprocessed output
        """
        raise NotImplementedError('Base class not implemented!')

    def get_number_of_output_channels(self) -> int:
        """ Abstract method for getting number of classes in the output layer

        Returns
        -------
        int
            Number of channels in the output layer"""
        raise NotImplementedError('Base class not implemented!')

    def check_loaded_model_outputs(self):
        """ Abstract method for checking if the model outputs are valid

        """
        raise NotImplementedError('Base class not implemented!')
