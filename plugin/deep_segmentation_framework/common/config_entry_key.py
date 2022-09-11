import enum

from qgis.core import QgsProject

from deep_segmentation_framework.common.defines import PLUGIN_NAME


class ConfigEntryKey(enum.Enum):
    """
    Entries to be stored in Project Configuration.
    Second element of enum value (in tuple) is the default value for this field
    """

    MODEL_FILE_PATH = enum.auto(), ''
    INPUT_LAYER_ID = enum.auto(), ''
    MODEL_TYPE = enum.auto(), ''
    PREPROCESSING_RESOLUTION = enum.auto(), 3.0
    PREPROCESSING_TILES_OVERLAP = enum.auto(), 15

    SEGMENTATION_PROBABILITY_THRESHOLD_ENABLED = enum.auto(), True
    SEGMENTATION_PROBABILITY_THRESHOLD_VALUE = enum.auto(), 0.5
    SEGMENTATION_REMOVE_SMALL_SEGMENT_ENABLED = enum.auto(), True
    SEGMENTATION_REMOVE_SMALL_SEGMENT_SIZE = enum.auto(), 9

    DETECTION_CONFIDENCE = enum.auto(), 0.5
    DETECTION_IOU = enum.auto(), 0.5

    DATA_EXPORT_DIR = enum.auto(), ''
    DATA_EXPORT_TILES_ENABLED = enum.auto(), True
    DATA_EXPORT_SEGMENTATION_MASK_ENABLED = enum.auto(), False
    DATA_EXPORT_SEGMENTATION_MASK_ID = enum.auto(), ''

    def get(self):
        """
        Get value store in config (or a default one)
        """
        read_function = None

        # check the default value to determine the entry type
        default_value = self.value[1]  # second element in the 'value' tuple
        if isinstance(default_value, int):
            read_function = QgsProject.instance().readNumEntry
        elif isinstance(default_value, float):
            read_function = QgsProject.instance().readDoubleEntry
        elif isinstance(default_value, bool):
            read_function = QgsProject.instance().readBoolEntry
        elif isinstance(default_value, str):
            read_function = QgsProject.instance().readEntry
        else:
            raise Exception("Unsupported entry type!")

        print(self.name)
        value, _ = read_function(PLUGIN_NAME, self.name, default_value)
        return value

    def set(self, value):
        write_function = None

        # check the default value to determine the entry type
        default_value = self.value[1]  # second element in the 'value' tuple
        if isinstance(default_value, int):
            write_function = QgsProject.instance().writeEntry
        elif isinstance(default_value, float):
            write_function = QgsProject.instance().writeEntryDouble
        elif isinstance(default_value, bool):
            write_function = QgsProject.instance().writeEntryBool
        elif isinstance(default_value, str):
            write_function = QgsProject.instance().writeEntry
        else:
            raise Exception("Unsupported entry type!")

        write_function(PLUGIN_NAME, self.name, value)
