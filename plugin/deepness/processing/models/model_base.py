import json
from typing import Optional

import numpy as np
import onnxruntime as ort


class ModelBase:
    """
    Wraps the ONNX model used during processing into a common interface
    """

    def __init__(self, model_file_path: str):
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
    def get_model_type_from_metadata(cls, model_file_path) -> Optional[str]:
        model = cls(model_file_path)
        return model.get_metadata_model_type()

    def get_input_shape(self):
        """
        Get shape of the input for the model
        """
        return self.input_shape

    def get_input_size_in_pixels(self):
        """
        Get number of input pixels in x and y direction (the same value)
        """
        return self.input_shape[-2:]

    def get_channel_name(self, channel_id: int) -> str:
        """
        Get channel name by id if exists in model metadata
        """
        meta = self.sess.get_modelmeta()
        channel_id_str = str(channel_id)
        default_return = f'channel_{channel_id_str}'

        if 'class_names' in meta.custom_metadata_map:
            class_names = json.loads(meta.custom_metadata_map['class_names'])

            return class_names.get(channel_id_str, default_return)
        else:
            return default_return

    def get_metadata_model_type(self) -> Optional[str]:
        meta = self.sess.get_modelmeta()
        name = 'model_type'
        if name in meta.custom_metadata_map:
            value = json.loads(meta.custom_metadata_map[name])
            return str(value).capitalize()
        return None

    def get_metadata_resolution(self) -> Optional[float]:
        meta = self.sess.get_modelmeta()
        name = 'resolution'
        if name in meta.custom_metadata_map:
            value = json.loads(meta.custom_metadata_map[name])
            return float(value)
        return None

    def get_metadata_tile_size(self) -> Optional[int]:
        meta = self.sess.get_modelmeta()
        name = 'tile_size'
        if name in meta.custom_metadata_map:
            value = json.loads(meta.custom_metadata_map[name])
            return int(value)
        return None

    def get_metadata_tiles_overlap(self) -> Optional[int]:
        meta = self.sess.get_modelmeta()
        name = 'tiles_overlap'
        if name in meta.custom_metadata_map:
            value = json.loads(meta.custom_metadata_map[name])
            return int(value)
        return None

    def get_metadata_segmentation_threshold(self) -> Optional[float]:
        meta = self.sess.get_modelmeta()
        name = 'seg_thresh'
        if name in meta.custom_metadata_map:
            value = json.loads(meta.custom_metadata_map[name])
            return float(value)
        return None

    def get_metadata_segmentation_small_segment(self) -> Optional[int]:
        meta = self.sess.get_modelmeta()
        name = 'seg_small_segment'
        if name in meta.custom_metadata_map:
            value = json.loads(meta.custom_metadata_map[name])
            return int(value)
        return None

    def get_metadata_regression_output_scaling(self) -> Optional[float]:
        meta = self.sess.get_modelmeta()
        name = 'reg_output_scaling'
        if name in meta.custom_metadata_map:
            value = json.loads(meta.custom_metadata_map[name])
            return float(value)
        return None

    def get_metadata_detection_confidence(self) -> Optional[float]:
        meta = self.sess.get_modelmeta()
        name = 'det_conf'
        if name in meta.custom_metadata_map:
            value = json.loads(meta.custom_metadata_map[name])
            return float(value)
        return None

    def get_metadata_detection_iou_threshold(self) -> Optional[float]:
        meta = self.sess.get_modelmeta()
        name = 'det_iou_thresh'
        if name in meta.custom_metadata_map:
            value = json.loads(meta.custom_metadata_map[name])
            return float(value)
        return None

    def get_number_of_channels(self):
        return self.input_shape[-3]

    def process(self, img):
        """
        Process a single tile image
        :param img: RGB img [TILE_SIZE x TILE_SIZE x channels], type uint8, values 0 to 255
        :return: single prediction
        """

        input_batch = self.preprocessing(img)
        model_output = self.sess.run(
            output_names=None,
            input_feed={self.input_name: input_batch})
        res = self.postprocessing(model_output)
        return res

    def preprocessing(self, img: np.ndarray):
        return NotImplementedError

    def postprocessing(self, outs: np.ndarray):
        return NotImplementedError

    def get_number_of_output_channels(self):
        return NotImplementedError

    def check_loaded_model_outputs(self):
        return NotImplementedError

