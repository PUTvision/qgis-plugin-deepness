import numpy as np

from deep_segmentation_framework.processing.models.base import BaseModel


class Segmentor(BaseModel):
    def __init__(self, model_file_path: str):
        super(Segmentor, self).__init__(model_file_path)

    def preprocessing(self, image: np.ndarray):
        img = image[:, :, :self.input_shape[-3]]

        input_batch = img.astype('float32')
        input_batch /= 255
        input_batch = input_batch.transpose(2, 0, 1)
        input_batch = np.expand_dims(input_batch, axis=0)

        return input_batch, {}

    def postprocessing(self, model_output):
        labels = np.clip(model_output[0][0], 0, 1)

        return labels

    @classmethod
    def get_class_display_name(cls):
        return cls.__name__
