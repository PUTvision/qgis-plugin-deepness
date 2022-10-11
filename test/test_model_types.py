from deepness.processing.models.model_types import ModelDefinition, ModelType


def test_model_types():
    # TODO - test does nothing
    a = ModelType.SEGMENTATION.value
    b = ModelType.SEGMENTATION.name

    model_type = ModelType(a)
    model_definition = ModelDefinition.get_definition_for_type(model_type)


if __name__ == '__main__':
    test_model_types()
    
