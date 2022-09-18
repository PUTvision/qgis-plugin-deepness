from deep_segmentation_framework.processing.models.model_base import ModelBase
from test.test_utils import get_dummy_segmentation_model_path


MODEL_FILE_PATH = get_dummy_segmentation_model_path()


def test_load_and_validate_metadata():
    model = ModelBase(model_file_path=MODEL_FILE_PATH)
    assert model.get_input_shape() == [1, 3, 512, 512]
    assert model.get_number_of_channels() == 3
    assert model.get_input_size_in_pixels() == [512, 512]


if __name__ == '__main__':
    test_load_and_validate_metadata()
