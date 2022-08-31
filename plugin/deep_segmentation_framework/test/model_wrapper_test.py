from deep_segmentation_framework.processing.model_wrapper import ModelWrapper


MODEL_FILE_PATH = '/home/przemek/Desktop/corn_segmentation_model.onnx'


def load_and_validate_metadata_test():
    model_wrapper = ModelWrapper(model_file_path=MODEL_FILE_PATH)
    assert model_wrapper.get_input_shape() == [1, 3, 512, 512]
    assert model_wrapper.get_number_of_channels() == 3
    assert model_wrapper.get_input_size_in_pixels() == [512, 512]


if __name__ == '__main__':
    load_and_validate_metadata_test()