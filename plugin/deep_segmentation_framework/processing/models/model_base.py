import json
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

