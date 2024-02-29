""" Module including the base model interfaces and utilities"""
import ast
import json
from typing import List, Optional

import numpy as np

from deepness.common.lazy_package_loader import LazyPackageLoader
from deepness.common.processing_parameters.standardization_parameters import StandardizationParameters

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
        self.standardization_parameters: StandardizationParameters = self.get_metadata_standarization_parameters()
        
        self.output_names = self.get_outputs_channel_names()

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

    def get_model_batch_size(self) -> Optional[int]:
        """ Get batch size of the model

        Returns
        -------
        Optional[int] | None
            Batch size or None if not found (dynamic batch size)
        """
        bs = self.input_shape[0]

        if isinstance(bs, str):
            return None
        else:
            return bs

    def get_input_size_in_pixels(self) -> int:
        """ Get number of input pixels in x and y direction (the same value)

        Returns
        -------
        int
            Number of pixels in x and y direction
        """
        return self.input_shape[-2:]

    def get_outputs_channel_names(self) -> Optional[List[List[str]]]:
        """ Get class names from metadata

        Returns
        -------
        List[List[str]] | None
            List of class names for each model output or None if not found
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

            if isinstance(class_names, dict):
                class_names = [class_names]

            sorted_by_key = [sorted(cn.items(), key=lambda kv: int(kv[0])) for cn in class_names]

            all_names = []
            
            for output_index in range(len(sorted_by_key)):
                output_names = []
                class_counter = 0
            
                for key, value in sorted_by_key[output_index]:
                    if int(key) != class_counter:
                        raise Exception("Class names in the model metadata are not consecutive (missing class label)")
                    class_counter += 1
                    output_names.append(value)
                all_names.append(output_names)

            return all_names

        return None

    def get_channel_name(self, layer_id: int, channel_id: int) -> str:
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
        
        channel_id_str = str(channel_id)
        default_return = f'channel_{channel_id_str}'

        if self.output_names is None:
            return default_return
        
        if layer_id >= len(self.output_names):
            raise Exception(f'Layer id {layer_id} is out of range of the model outputs')
        
        if channel_id >= len(self.output_names[layer_id]):
            raise Exception(f'Channel id {channel_id} is out of range of the model outputs')
        
        return f'{self.output_names[layer_id][channel_id]}'

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

    def get_metadata_standarization_parameters(self) -> Optional[StandardizationParameters]:
        """ Get standardization parameters from metadata if exists

        Returns
        -------
        Optional[StandardizationParameters]
            Standardization parameters or None if not found
        """
        meta = self.sess.get_modelmeta()
        name_mean = 'standardization_mean'
        name_std = 'standardization_std'

        param = StandardizationParameters(channels_number=self.get_input_shape()[-3])

        if name_mean in meta.custom_metadata_map and name_std in meta.custom_metadata_map:
            mean = json.loads(meta.custom_metadata_map[name_mean])
            std = json.loads(meta.custom_metadata_map[name_std])

            mean = [float(x) for x in mean]
            std = [float(x) for x in std]

            param.set_mean_std(mean=mean, std=std)

            return param

        return param  # default, no standardization

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

    def get_number_of_channels(self) -> int:
        """ Returns number of channels in the input layer

        Returns
        -------
        int
            Number of channels in the input layer
        """
        return self.input_shape[-3]

    def process(self, tiles_batched: np.ndarray):
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
        input_batch = self.preprocessing(tiles_batched)
        model_output = self.sess.run(
            output_names=None,
            input_feed={self.input_name: input_batch})
        res = self.postprocessing(model_output)
        return res

    def preprocessing(self, tiles_batched: np.ndarray) -> np.ndarray:
        """ Preprocess the batch of images for the model (resize, normalization, etc)

        Parameters
        ----------
        image : np.ndarray
            Batch of images to preprocess (N,H,W,C), RGB, 0-255

        Returns
        -------
        np.ndarray
            Preprocessed batch of image (N,C,H,W), RGB, 0-1
        """

        # imported here, to avoid isseue with uninstalled dependencies during the first plugin start
        # in other places we use LazyPackageLoader, but here it is not so easy
        import deepness.processing.models.preprocessing_utils as preprocessing_utils

        tiles_batched = preprocessing_utils.limit_channels_number(tiles_batched, limit=self.input_shape[-3])
        tiles_batched = preprocessing_utils.normalize_values_to_01(tiles_batched)
        tiles_batched = preprocessing_utils.standardize_values(tiles_batched, params=self.standardization_parameters)
        tiles_batched = preprocessing_utils.transpose_nhwc_to_nchw(tiles_batched)

        return tiles_batched

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

    def get_number_of_output_channels(self) -> List[int]:
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
