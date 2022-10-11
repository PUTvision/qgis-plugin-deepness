import numpy as np

from deepness.processing.models.model_base import ModelBase


class Segmentor(ModelBase):
    def __init__(self, model_file_path: str):
        super(Segmentor, self).__init__(model_file_path)

    def preprocessing(self, image: np.ndarray):
        img = image[:, :, :self.input_shape[-3]]

        input_batch = img.astype('float32')
        input_batch /= 255
        input_batch = input_batch.transpose(2, 0, 1)
        input_batch = np.expand_dims(input_batch, axis=0)

        return input_batch

    def postprocessing(self, model_output):
        labels = np.clip(model_output[0][0], 0, 1)

        return labels

    def get_number_of_output_channels(self):
        if len(self.outputs_layers) == 1:
            return self.outputs_layers[0].shape[-3]
        else:
            return NotImplementedError

    @classmethod
    def get_class_display_name(cls):
        return cls.__name__

    def check_loaded_model_outputs(self):
        if len(self.outputs_layers) == 1:
            shape = self.outputs_layers[0].shape

            if len(shape) != 4:
                raise Exception(f'Segmentation model output should have 4 dimensions: (B,C,H,W). Has {shape}')

            if shape[0] != 1:
                raise Exception(f'Segmentation model can handle only 1-Batch outputs. Has {shape}')

            if shape[2] != shape[3]:
                raise Exception(f'Segmentation model can handle only square outputs masks. Has: {shape}')
            
        else:
            raise NotImplementedError
