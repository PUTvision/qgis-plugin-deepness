"""
This file contains a single widget, which is embedded in the main dockwiget - to select the training data export parameters
"""

import os

from qgis.PyQt import QtWidgets, uic
from qgis.PyQt.QtWidgets import QFileDialog
from qgis.core import QgsMapLayerProxyModel
from qgis.core import QgsProject

from deepness.common.config_entry_key import ConfigEntryKey
from deepness.common.processing_parameters.map_processing_parameters import MapProcessingParameters
from deepness.common.processing_parameters.training_data_export_parameters import \
    TrainingDataExportParameters

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'training_data_export_widget.ui'))


class TrainingDataExportWidget(QtWidgets.QWidget, FORM_CLASS):
    """
    Widget responsible for defining the parameters for the Trainign Data Export process (not doing the actual export).

    UI design defined in the `training_data_export_widget.ui` file.
    """

    def __init__(self, rlayer, parent=None):
        super(TrainingDataExportWidget, self).__init__(parent)
        self.setupUi(self)
        self._create_connections()
        self._setup_misc_ui()

    def load_ui_from_config(self):
        layers = QgsProject.instance().mapLayers()

        self.lineEdit_outputDirPath.setText(ConfigEntryKey.DATA_EXPORT_DIR.get())
        self.checkBox_exportImageTiles.setChecked(
            ConfigEntryKey.DATA_EXPORT_TILES_ENABLED.get())
        self.checkBox_exportMaskEnabled.setChecked(
            ConfigEntryKey.DATA_EXPORT_SEGMENTATION_MASK_ENABLED.get())

        segmentation_layer_id = ConfigEntryKey.DATA_EXPORT_SEGMENTATION_MASK_ID.get()
        if segmentation_layer_id and segmentation_layer_id in layers:
            self.mMapLayerComboBox_inputLayer.setLayer(layers[segmentation_layer_id])

    def save_ui_to_config(self):
        ConfigEntryKey.DATA_EXPORT_DIR.set(self.lineEdit_outputDirPath.text())
        ConfigEntryKey.DATA_EXPORT_TILES_ENABLED.set(
            self.checkBox_exportImageTiles.isChecked())
        ConfigEntryKey.DATA_EXPORT_SEGMENTATION_MASK_ENABLED.set(
            self.checkBox_exportMaskEnabled.isChecked())
        ConfigEntryKey.DATA_EXPORT_SEGMENTATION_MASK_ID.set(self.get_segmentation_mask_layer_id())

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
        """ Get the parameters from the UI for the data exporting process"""
        if self.checkBox_exportMaskEnabled.isChecked():
            segmentation_mask_layer_id = self.mMapLayerComboBox_maskLayer.currentLayer().id()
        else:
            segmentation_mask_layer_id = None

        params = TrainingDataExportParameters(
            **map_processing_parameters.__dict__,
            export_image_tiles=self.checkBox_exportImageTiles.isChecked(),
            segmentation_mask_layer_id=segmentation_mask_layer_id,
            output_directory_path=self.lineEdit_outputDirPath.text(),
        )
        return params
