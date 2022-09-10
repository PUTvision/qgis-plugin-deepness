from deep_segmentation_framework.processing.models.model_types import ModelDefinition, ModelType


def main():
    a = ModelType.SEGMENTATION.value
    b = ModelType.SEGMENTATION.name
    model_type_txt = 'SEGMENTATION'
    model_type = ModelType(model_type_txt)
    model_definition = ModelDefinition.get_definition_for_type(model_type)


if __name__ == '__main__':
    main()