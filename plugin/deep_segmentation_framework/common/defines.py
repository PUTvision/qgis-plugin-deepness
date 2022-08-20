import enum

from qgis.core import QgsProject

PLUGIN_NAME = 'DeepSegmentationFramework'
LOG_TAB_NAME = PLUGIN_NAME

IS_DEBUG = True  # enable some debugging options  TODO set from environemnt variable


class ConfigEntryKey(enum.Enum):
    """
    Entries to be stored in Project Configuration.
    Value assigned to each enum is a default value.
    """

    MODEL_FILE_PATH = ''

    def get(self):
        read_function = None

        # check the default value to determine the entry type
        if isinstance(self.value, int):
            read_function = QgsProject.instance().readNumEntry
        elif isinstance(self.value, float):
            read_function = QgsProject.instance().readDoubleEntry
        elif isinstance(self.value, bool):
            read_function = QgsProject.instance().readBoolEntry
        elif isinstance(self.value, str):
            read_function = QgsProject.instance().readEntry
        else:
            raise Exception("Unsupported entry type!")

        value, _ = read_function(PLUGIN_NAME, self.name, self.value)
        return value

    def set(self, value):
        QgsProject.instance().writeEntry(PLUGIN_NAME, self.name, value)
