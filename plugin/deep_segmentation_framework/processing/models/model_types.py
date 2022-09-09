from deep_segmentation_framework.processing.models.detector import Detector
from deep_segmentation_framework.processing.models.segmentor import Segmentor


MODEL_TYPES_NAMES = [
    Segmentor.get_class_display_name(),
    Detector.get_class_display_name(),
]