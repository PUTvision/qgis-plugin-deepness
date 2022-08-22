import numpy as np
import onnxruntime as ort


class ModelWrapper:
    """
    Wraps the ONNX model used during processing into a common interface
    """

    def __init__(self, model_file_path):
        self.model_file_path = model_file_path
        self.sess = ort.InferenceSession(self.model_file_path)
        inputs = self.sess.get_inputs()
        if len(inputs) > 1:
            raise Exception("ONNX model: unsupported number of inputs")
        input_0 = inputs[0]
        self.output_0_name = self.sess.get_outputs()[0].name  # We expect only the first output
        self.input_shape = input_0.shape
        self.input_name = input_0.name

    def get_input_size_in_pixels(self):
        """
        Get number of input pixels in x and y direction (the same value)
        """
    todo return shape

    def get_number_of_channels(self):
        return self.input_shape[-3]

    def process(self, img):
        """

        :param img: RGB img [TILE_SIZE x TILE_SIZE x channels], type uint8, values 0 to 255
        :return: single prediction mask
        """

        # TODO add support for channels mapping

        img = img[:, :, :self.get_number_of_channels()]

        input_batch = img.astype('float32')
        input_batch /= 255
        input_batch = input_batch.transpose(2, 0, 1)
        input_batch = np.expand_dims(input_batch, axis=0)

        model_output = self.sess.run(
            output_names=[self.output_0_name],
            input_feed={self.input_name: input_batch})

        # TODO - add support for multiple output classes. For now just take 0 layer
        damaged_area_onnx = model_output[0][0][1] * 255
        return damaged_area_onnx
