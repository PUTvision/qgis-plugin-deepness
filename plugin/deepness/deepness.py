import os
import traceback

from qgis.PyQt.QtWidgets import QMessageBox

from deepness.common.processing_parameters.map_processing_parameters import MapProcessingParameters, ProcessedAreaType
from deepness.common.processing_parameters.training_data_export_parameters import TrainingDataExportParameters
from deepness.images.get_image_path import get_icon_path
from deepness.processing.map_processor.map_processor_training_data_export import MapProcessorTrainingDataExport
from deepness.processing.map_processor.map_processing_result import MapProcessingResult, MapProcessingResultFailed, \
    MapProcessingResultCanceled, MapProcessingResultSuccess
from deepness.processing.models.model_types import ModelDefinition

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()  # increase limit of pixels (2^30), before importing cv2
import cv2

from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication, Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction
from qgis.core import QgsVectorLayer
from qgis.core import QgsApplication
from qgis.core import QgsProject
from qgis.gui import QgisInterface
from qgis.core import Qgis

# Initialize Qt resources from file resources.py
from deepness.common.defines import PLUGIN_NAME, IS_DEBUG

# Import the code for the DockWidget
from deepness.deepness_dockwidget import DeepnessDockWidget
import os.path


class Deepness:
    """QGIS Plugin Implementation."""

    def __init__(self, iface: QgisInterface):
        """
        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        print('1) __init__')
        # Save reference to the QGIS interface
        self.iface = iface

        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)

        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'Deepness_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            QCoreApplication.installTranslator(self.translator)

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&Deep Segmentation Framework')
        # TODO: We are going to let the user set this up in a future iteration
        self.toolbar = self.iface.addToolBar(u'Deepness')
        self.toolbar.setObjectName(u'Deepness')

        self.pluginIsActive = False
        self.dockwidget = None
        self._map_processor = None

    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('Deepness', message)

    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.toolbar.addAction(action)

        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""
        print('2) initGui')

        icon_path = get_icon_path()
        self.add_action(
            icon_path,
            text=self.tr(u'Deep Segmentation Framework'),
            callback=self.run,
            parent=self.iface.mainWindow())

        if IS_DEBUG:
            self.run()

    def onClosePlugin(self):
        """Cleanup necessary items here when plugin dockwidget is closed"""
        print('3) onClosePlugin')

        # disconnects
        self.dockwidget.closingPlugin.disconnect(self.onClosePlugin)

        # remove this statement if dockwidget is to remain
        # for reuse if plugin is reopened
        # Commented next statement since it causes QGIS crashe
        # when closing the docked window:
        # self.dockwidget = None

        self.pluginIsActive = False

    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        print('4) unload')

        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&Deep Segmentation Framework'),
                action)
            self.iface.removeToolBarIcon(action)
        # remove the toolbar
        del self.toolbar

    def _layers_changed(self, _):
        pass

    def run(self):
        """Run method that loads and starts the plugin"""
        print(f'5) run. {self.dockwidget = }')

        if not self.pluginIsActive:
            self.pluginIsActive = True

            # dockwidget may not exist if:
            #    first run of plugin
            #    removed on close (see self.onClosePlugin method)
            if self.dockwidget is None:
                # Create the dockwidget (after translation) and keep reference
                self.dockwidget = DeepnessDockWidget(self.iface)
                self._layers_changed(None)
                QgsProject.instance().layersAdded.connect(self._layers_changed)
                QgsProject.instance().layersRemoved.connect(self._layers_changed)

            # connect to provide cleanup on closing of dockwidget
            self.dockwidget.closingPlugin.connect(self.onClosePlugin)
            self.dockwidget.run_model_inference_signal.connect(self._run_model_inference)
            self.dockwidget.run_training_data_export_signal.connect(self._run_training_data_export)

            self.iface.addDockWidget(Qt.RightDockWidgetArea, self.dockwidget)
            self.dockwidget.show()

    def _are_map_processing_parameters_are_correct(self, params: MapProcessingParameters):
        if self._map_processor and self._map_processor.is_busy():
            msg = "Error! Processing already in progress! Please wait or cancel previous task."
            self.iface.messageBar().pushMessage(PLUGIN_NAME, msg, level=Qgis.Critical, duration=7)
            return False

        rlayer = QgsProject.instance().mapLayers()[params.input_layer_id]
        if rlayer is None:
            msg = "Error! Please select the layer to process first!"
            self.iface.messageBar().pushMessage(PLUGIN_NAME, msg, level=Qgis.Critical, duration=7)
            return False

        if isinstance(rlayer, QgsVectorLayer):
            msg = "Error! Please select a raster layer (vector layer selected)"
            self.iface.messageBar().pushMessage(PLUGIN_NAME, msg, level=Qgis.Critical, duration=7)
            return False

        return True

    def _display_processing_started_info(self):
        msg = "Error! Please select the layer to process first!"
        self.iface.messageBar().pushMessage(PLUGIN_NAME, msg, level=Qgis.Info, duration=2)

    def _run_training_data_export(self, training_data_export_parameters: TrainingDataExportParameters):
        if not self._are_map_processing_parameters_are_correct(training_data_export_parameters):
            return

        vlayer = None

        rlayer = QgsProject.instance().mapLayers()[training_data_export_parameters.input_layer_id]
        if training_data_export_parameters.processed_area_type == ProcessedAreaType.FROM_POLYGONS:
            vlayer = QgsProject.instance().mapLayers()[training_data_export_parameters.mask_layer_id]

        self._map_processor = MapProcessorTrainingDataExport(
            rlayer=rlayer,
            vlayer_mask=vlayer,
            map_canvas=self.iface.mapCanvas(),
            params=training_data_export_parameters)
        self._map_processor.finished_signal.connect(self._map_processor_finished)
        self._map_processor.show_img_signal.connect(self._show_img)
        QgsApplication.taskManager().addTask(self._map_processor)
        self._display_processing_started_info()

    def _run_model_inference(self, params: MapProcessingParameters):
        if not self._are_map_processing_parameters_are_correct(params):
            return

        vlayer = None

        rlayer = QgsProject.instance().mapLayers()[params.input_layer_id]
        if params.processed_area_type == ProcessedAreaType.FROM_POLYGONS:
            vlayer = QgsProject.instance().mapLayers()[params.mask_layer_id]

        model_definition = ModelDefinition.get_definition_for_params(params)
        map_processor_class = model_definition.map_processor_class

        self._map_processor = map_processor_class(
            rlayer=rlayer,
            vlayer_mask=vlayer,
            map_canvas=self.iface.mapCanvas(),
            params=params)
        self._map_processor.finished_signal.connect(self._map_processor_finished)
        self._map_processor.show_img_signal.connect(self._show_img)
        QgsApplication.taskManager().addTask(self._map_processor)
        self._display_processing_started_info()

    @staticmethod
    def _show_img(img_rgb, window_name):
        img_bgr = img_rgb[..., ::-1]
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 800)
        cv2.imshow(window_name, img_bgr)
        cv2.waitKey(1)

    def _map_processor_finished(self, result: MapProcessingResult):
        if isinstance(result, MapProcessingResultFailed):
            msg = f'Error! Processing error: "{result.message}"!'
            self.iface.messageBar().pushMessage(PLUGIN_NAME, msg, level=Qgis.Critical, duration=14)
            if result.exception is not None:
                trace = '\n'.join(traceback.format_tb(result.exception.__traceback__)[-1:])
                msg = f'{msg}\n\n\n' \
                      f'Details: ' \
                      f'{str(result.exception.__class__.__name__)} - {result.exception}\n' \
                      f'Last Traceback: \n' \
                      f'{trace}'
                QMessageBox.critical(self.dockwidget, "Unhandled exception", msg)
        elif isinstance(result, MapProcessingResultCanceled):
            msg = f'Info! Processing canceled by user!'
            self.iface.messageBar().pushMessage(PLUGIN_NAME, msg, level=Qgis.Info, duration=7)
        elif isinstance(result, MapProcessingResultSuccess):
            msg = 'Processing finished!'
            self.iface.messageBar().pushMessage(PLUGIN_NAME, msg, level=Qgis.Success, duration=3)
            message_to_show = result.message
            QMessageBox.information(self.dockwidget, "Processing Result", message_to_show)
        self._map_processor = None
