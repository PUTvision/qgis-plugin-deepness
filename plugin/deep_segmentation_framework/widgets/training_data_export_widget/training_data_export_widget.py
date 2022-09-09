import os

from PyQt5.QtWidgets import QFileDialog

from qgis.PyQt import QtWidgets, uic
from qgis.core import QgsMapLayerProxyModel
from qgis.PyQt.QtWidgets import QFileDialog

from deep_segmentation_framework.common.channels_mapping import ChannelsMapping
from deep_segmentation_framework.common.processing_parameters.map_processing_parameters import MapProcessingParameters
from deep_segmentation_framework.common.processing_parameters.training_data_export_parameters import \
    TrainingDataExportParameters

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'training_data_export_widget.ui'))


class TrainingDataExportWidget(QtWidgets.QWidget, FORM_CLASS):
    """
    Widget responsible for exporting .
    """

    def __init__(self, rlayer, parent=None):
        super(TrainingDataExportWidget, self).__init__(parent)
        self.setupUi(self)
        self._create_connections()
        self._setup_misc_ui()

    def get_channels_mapping(self) -> ChannelsMapping:
        if self.radioButton_defaultMapping.isChecked():
            return self._channels_mapping.get_as_default_mapping()
        else:  # advanced mapping
            return self._channels_mapping

    def _browse_output_directory(self):
        current_directory = self.lineEdit_outputDirPath.text()
        if not current_directory:
            current_directory = os.path.expanduser('~')
        new_directory = QFileDialog.getExistingDirectory(
            self,
            "Select Directory",
            current_directory,
            QFileDialog.ShowDirsOnly)
        self.lineEdit_outputDirPath.setText(new_directory)

    def _enable_disable_mask_layer_selection(self):
        is_enabled = self.checkBox_exportMaskEnabled.isChecked()
        self.mMapLayerComboBox_maskLayer.setEnabled(is_enabled)

    def _create_connections(self):
        self.checkBox_exportMaskEnabled.toggled.connect(self._enable_disable_mask_layer_selection)
        self.pushButton_browseOutputDirectory.clicked.connect(self._browse_output_directory)

    def _setup_misc_ui(self):
        self.mMapLayerComboBox_maskLayer.setFilters(QgsMapLayerProxyModel.VectorLayer)
        self._enable_disable_mask_layer_selection()

    def get_segmentation_mask_layer_id(self):
        if not self.checkBox_exportMaskEnabled.isChecked():
            return None
        return self.mMapLayerComboBox_maskLayer.currentLayer().id()

    def get_training_data_export_parameters(self, map_processing_parameters: MapProcessingParameters):
        params = TrainingDataExportParameters(
            **map_processing_parameters.__dict__,
            export_image_tiles=self.checkBox_exportImageTiles.isChecked(),
            segmentation_mask_layer_id=self.mMapLayerComboBox_maskLayer.currentLayer().id(),
            output_directory_path=self.lineEdit_outputDirPath.text(),
        )
        return params
