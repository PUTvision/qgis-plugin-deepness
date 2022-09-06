import os
from typing import Optional

from qgis.PyQt.QtWidgets import QLabel

from qgis.PyQt.QtWidgets import QComboBox
from qgis.PyQt import QtWidgets, uic
from qgis.core import Qgis

from deep_segmentation_framework.common.channels_mapping import ChannelsMapping, ImageChannelStandaloneBand, \
    ImageChannelCompositeByte
from deep_segmentation_framework.common.channels_mapping import ImageChannel
from deep_segmentation_framework.processing.model_wrapper import ModelWrapper

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'input_channels_mapping_widget.ui'))


class InputChannelsMappingWidget(QtWidgets.QWidget, FORM_CLASS):
    """
    Widget responsible for mapping image channels to model input channels.
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

    def get_channels_mapping(self) -> ChannelsMapping:
        if self.radioButton_defaultMapping.isChecked():
            return self._channels_mapping.get_as_default_mapping()
        else:  # advanced mapping
            return self._channels_mapping

    def get_channels_mapping_for_training_data_export(self) -> ChannelsMapping:
        mapping = self._channels_mapping.get_as_default_mapping()
        mapping.set_number_of_model_inputs_same_as_image_channels()
        return mapping

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

        image_channels = []  # type: List[ImageChannel]

        if number_of_image_bands == 1:
            # if there is one band, then there is probably more "bands" hidden in a more complex data type (e.g. RGBA)
            data_type = rlayer.dataProvider().dataType(1)
            if data_type == Qgis.DataType.Byte:
                image_channel = ImageChannelStandaloneBand(
                    band_number=1,
                    name=rlayer.bandName(1))
                image_channels.append(image_channel)
            elif data_type == Qgis.DataType.ARGB32:
                # Alpha channel is at byte number 3, red is byte 2, ... - reversed order
                band_names = [
                    'Alpha (band 4)',
                    'Red (band 1)',
                    'Green (band 2)',
                    'Blue (band 3)',
                ]
                for i in [1, 2, 3, 0]:  # We want order of model inputs as 'RGB' first and then 'A'
                    image_channel = ImageChannelCompositeByte(
                        byte_number=3 - i,  # bytes are in reversed order
                        name=band_names[i])
                    image_channels.append(image_channel)
            else:
                raise Exception("Invalid input layer data type!")
        else:
            for band_number in range(1, number_of_image_bands + 1):  # counted from 1
                image_channel = ImageChannelStandaloneBand(
                    band_number=band_number,
                    name=rlayer.bandName(band_number))
                image_channels.append(image_channel)

        self.label_imageInputs.setText(f'{len(image_channels)}')
        self._channels_mapping.set_image_channels(image_channels)
        self.regenerate_mapping()

    def _combobox_index_changed(self, model_input_channel_number):
        combobox = self._channels_mapping_comboboxes[model_input_channel_number]  # type: QComboBox
        image_channel_index = combobox.currentIndex()
        # print(f'Combobox {model_input_channel_number} changed to {current_index}')
        self._channels_mapping.set_image_channel_for_model_input(
            model_input_number=model_input_channel_number,
            image_channel_index=image_channel_index)

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

            if self._channels_mapping.get_number_of_image_channels() > 0:
                # image channel witch is currently assigned to the current model channel
                image_channel_index = self._channels_mapping.get_image_channel_index_for_model_input(
                    model_input_channel_number)
                combobox.setCurrentIndex(image_channel_index)

            combobox.currentIndexChanged.connect(
                lambda _, v=model_input_channel_number: self._combobox_index_changed(v))

            self.gridLayout_mapping.addWidget(label, model_input_channel_number, 0)
            self.gridLayout_mapping.addWidget(combobox, model_input_channel_number, 1, 1, 2)

            self._channels_mapping_comboboxes.append(combobox)
            self._channels_mapping_labels.append(label)
