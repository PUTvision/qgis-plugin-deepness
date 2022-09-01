import enum
import logging
import os
from dataclasses import dataclass
from typing import Optional

import onnxruntime as ort
from qgis.PyQt.QtWidgets import QLabel
from qgis.PyQt.QtWidgets import QMessageBox

from qgis.PyQt.QtWidgets import QComboBox
from qgis.PyQt import QtGui, QtWidgets, uic
from qgis.PyQt.QtCore import pyqtSignal
from qgis.core import QgsMapLayerProxyModel
from qgis.core import QgsVectorLayer
from qgis.core import QgsRasterLayer
from qgis.core import QgsMessageLog
from qgis.core import QgsProject
from qgis.core import QgsVectorLayer
from qgis.core import Qgis
from qgis.PyQt.QtWidgets import QInputDialog, QLineEdit, QFileDialog

from deep_segmentation_framework.common.channels_mapping import ChannelsMapping, ImageChannelStandaloneBand, \
    ImageChannelCompositeByte
from deep_segmentation_framework.common.channels_mapping import ImageChannel
from deep_segmentation_framework.common.defines import PLUGIN_NAME, LOG_TAB_NAME, ConfigEntryKey
from deep_segmentation_framework.common.inference_parameters import InferenceParameters, ProcessedAreaType
from deep_segmentation_framework.processing.model_wrapper import ModelWrapper

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'input_channels_mapping_widget.ui'))


class InputChannelsMappingWidget(QtWidgets.QWidget, FORM_CLASS):
    """
    Widget responsible for mapping image channels to model input channels.
    Can be skipped for the defau
    """

    def __init__(self, rlayer, parent=None):
        super(InputChannelsMappingWidget, self).__init__(parent)
        self.setupUi(self)
        self._model_wrapper = None  # type: Optional[ModelWrapper]
        self._rlayer = rlayer  # type: QgsRasteLayer
        self._create_connections()
        self._channels_mapping = ChannelsMapping()

        self._channels_mapping_labels = []  # type: List[QLabel]
        self._channels_mapping_comboboxes = []  # type: List[QComboBox]

        self._selection_mode_changed()

    def _create_connections(self):
        self.radioButton_defaultMapping.clicked.connect(self._selection_mode_changed)
        self.radioButton_advancedMapping.clicked.connect(self._selection_mode_changed)

    def _selection_mode_changed(self):
        is_advanced = self.radioButton_advancedMapping.isChecked()
        self.widget_mapping.setVisible(is_advanced)

    def set_model(self, model_wrapper: ModelWrapper):
        self._model_wrapper = model_wrapper
        number_of_channels = self._model_wrapper.get_number_of_channels()
        self.label_modelInputs.setText(f'{number_of_channels}')
        self._channels_mapping.set_number_of_model_inputs(number_of_channels)
        self.regenerate_mapping()

    def set_rlayer(self, rlayer):
        self._rlayer = rlayer
        number_of_image_bands = rlayer.bandCount()
        self.label_imageInputs.setText(f'{number_of_image_bands}')

        image_channels = []  # type: List[ImageChannel]

        if number_of_image_bands == 1:
            # if there is one band, then there is probably more "bands" hidden in a more complex data type (e.g. RGBA)
            data_type = rlayer.dataProvider().dataType(1)
            if data_type == Qgis.DataType.Byte:
                image_channel = ImageChannelStandaloneBand(
                    band_no=1,
                    name=rlayer.bandName(1))
                image_channels.append(image_channel)
            elif data_type == Qgis.DataType.ARGB32:
                band_names = [
                    'Alpha (band 4)',
                    'Red (band 1)',
                    'Green (band 2)',
                    'Blue (band 3)',
                ]
                for i in range(1, 4):  # RGB bytes, without A
                    image_channel = ImageChannelCompositeByte(
                        byte_number=i,
                        name=band_names[i])
                    image_channels.append(image_channel)
                image_channel = ImageChannelCompositeByte(
                    byte_number=0,
                    name=band_names[0])
                image_channels.append(image_channel)
            else:
                raise Exception("Invalid input layer data type!")
        else:
            for band_no in range(1, number_of_image_bands + 1):  # counted from 1
                image_channel = ImageChannelStandaloneBand(
                    band_no=band_no,
                    name=rlayer.bandName(band_no))
                image_channels.append(image_channel)

        self._channels_mapping.set_image_channels(image_channels)
        self.regenerate_mapping()

    def regenerate_mapping(self):
        for combobox in self._channels_mapping_comboboxes:
            self.gridLayout_mapping.removeWidget(combobox)
        self._channels_mapping_comboboxes.clear()
        for label in self._channels_mapping_labels:
            self.gridLayout_mapping.removeWidget(label)
        self._channels_mapping_labels.clear()

        for model_input_channel_number in range(self._channels_mapping.get_number_of_model_inputs()):
            label = QLabel(self)
            label.setText(f"Model input {model_input_channel_number}:")
            combobox = QComboBox(self)
            for image_channel in self._channels_mapping.get_image_channels():
                combobox.addItem(image_channel.name)

            # image channel witch is currently assigned to the current model channel
            image_channel_index = self._channels_mapping.get_image_channel_index_for_model_input(
                model_input_channel_number)
            combobox.setCurrentIndex(image_channel_index)

            self.gridLayout_mapping.addWidget(label, model_input_channel_number, 0)
            self.gridLayout_mapping.addWidget(combobox, model_input_channel_number, 1, 1, 2)

            self._channels_mapping_comboboxes.append(combobox)
            self._channels_mapping_labels.append(label)
