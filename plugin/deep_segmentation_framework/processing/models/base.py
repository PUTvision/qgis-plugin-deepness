import numpy as np
import onnxruntime as ort


class BaseModel:
    def __init__(self, model_file_path: str):
        self.model_file_path = model_file_path
        self.sess = ort.InferenceSession(self.model_file_path)
        inputs = self.sess.get_inputs()
        if len(inputs) > 1:
            raise Exception("ONNX model: unsupported number of inputs")
        input_0 = inputs[0]

        self.input_shape = input_0.shape
        self.input_name = input_0.name

        self.outputs_layers = self.sess.get_outputs()

    def preprocessing(self, img: np.ndarray):
        raise NotImplementedError

    def postprocessing(self, outs: np.ndarray):
        raise NotImplementedError

    def predict(self, img: np.ndarray):
        assert img.dtype == np.uint8
        assert len(img.shape) == 3

        input_batch = self.preprocessing(img)

        assert input_batch.dtype == np.float32
        assert len(input_batch.shape) == 4

        model_output = self.sess.run(
                output_names=None,
                input_feed={self.input_name: input_batch})

        return self.postprocessing(model_output)

