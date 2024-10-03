import enum
from dataclasses import dataclass

from deepness.common.processing_parameters.map_processing_parameters import MapProcessingParameters
from deepness.processing.models.model_base import ModelBase


@dataclass
class DetectorTypeParameters:
    """
    Defines some model-specific parameters for each model 'type' (e.g. default YOLOv7 has different model output shape than the one trained with Ultralytics' YOLO)
    """
    has_inverted_output_shape: bool = False  # whether the output shape of the model is inverted (and we need to apply np.transpose(model_output, (1, 0)))
    skipped_objectness_probability: bool = False  # whether the model output has has no 'objectness' probability, and only probability for each class
    ignore_objectness_probability: bool = False  # if the model output has the 'objectness' probability, we can still ignore it (it is needeed sometimes, when the probability was always 1...). The behavior should be the same as with `skipped_objectness_probability` (of course one model output needs be skipped)


class DetectorType(enum.Enum):
    """ Type of the detector model """

    YOLO_v5_v7_DEFAULT = 'YOLO_v5_or_v7_default'
    YOLO_v6 = 'YOLO_v6'
    YOLO_v9 = 'YOLO_v9'
    YOLO_ULTRALYTICS = 'YOLO_Ultralytics'
    YOLO_ULTRALYTICS_SEGMENTATION = 'YOLO_Ultralytics_segmentation'
    YOLO_ULTRALYTICS_OBB = 'YOLO_Ultralytics_obb'

    def get_parameters(self):
        if self == DetectorType.YOLO_v5_v7_DEFAULT:
            return DetectorTypeParameters()  # all default
        elif self == DetectorType.YOLO_v6:
            return DetectorTypeParameters(
                ignore_objectness_probability=True,
            )
        elif self == DetectorType.YOLO_v9:
            return DetectorTypeParameters(
                has_inverted_output_shape=True,
                skipped_objectness_probability=True,
            )
        elif self == DetectorType.YOLO_ULTRALYTICS or self == DetectorType.YOLO_ULTRALYTICS_SEGMENTATION or self == DetectorType.YOLO_ULTRALYTICS_OBB:
            return DetectorTypeParameters(
                has_inverted_output_shape=True,
                skipped_objectness_probability=True,
            )
        else:
            raise ValueError(f'Unknown detector type: {self}')

    def get_formatted_description(self):
        txt = ''
        txt += ' ' * 10 + f'Inverted output shape: {self.get_parameters().has_inverted_output_shape}\n'
        txt += ' ' * 10 + f'Skipped objectness : {self.get_parameters().skipped_objectness_probability}\n'
        txt += ' ' * 10 + f'Ignore objectness: {self.get_parameters().ignore_objectness_probability}\n'
        return txt

    def get_all_display_values():
        return [x.value for x in DetectorType]


@dataclass
class DetectionParameters(MapProcessingParameters):
    """
    Parameters for Inference of detection model (including pre/post-processing) obtained from UI.
    """

    model: ModelBase  # wrapper of the loaded model

    confidence: float
    iou_threshold: float

    detector_type: DetectorType = DetectorType.YOLO_v5_v7_DEFAULT  # parameters specific for each model type
