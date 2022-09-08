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
        self.output_0_name = self.sess.get_outputs()[0].name  # We expect only the first output
        self.input_shape = input_0.shape
        self.input_name = input_0.name

    def preprocessing(self, img: np.ndarray):
        raise NotImplementedError

    def postprocessing(self, outs: np.ndarray):
        raise NotImplementedError

    def predict(self, img: np.ndarray):
        assert img.dtype == np.uint8
        assert len(img.shape) == 3

        input_batch, kwargs = self.preprocessing(img)

        assert input_batch.dtype == np.float32
        assert type(kwargs) == dict
        assert len(input_batch.shape) == 4

        model_output = self.sess.run(
                output_names=[self.output_0_name],
                input_feed={self.input_name: input_batch})

        return self.postprocessing(model_output, **kwargs)

