# -*- coding: utf-8 -*-
"""
/***************************************************************************
 DeepSegmentationFramework
                                 A QGIS plugin
 This plugin allows to perform segmentation with deep neural networks
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2022-08-11
        git sha              : $Format:%H$
        copyright            : (C) 2022 by Przemyslaw Aszkowski
        email                : przemyslaw.aszkowski@gmail.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
import copy
import time

import numpy as np
import cv2

from PyQt5.QtCore import QByteArray
from PyQt5.QtGui import QPixmap
from qgis.PyQt.QtCore import pyqtSignal
from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication, Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction
from qgis.core import QgsVectorLayer
from qgis.core import QgsUnitTypes
from qgis.core import QgsRectangle
from qgis.core import QgsMessageLog
from qgis.core import QgsApplication
from qgis.core import QgsTask
from qgis.core import QgsProject
from qgis.core import QgsCoordinateTransform
from qgis.gui import QgisInterface
from qgis.core import Qgis
import qgis

# Initialize Qt resources from file resources.py
from deep_segmentation_framework.common.defines import PLUGIN_NAME, LOG_TAB_NAME
from deep_segmentation_framework.common.inference_parameters import InferenceParameters
from deep_segmentation_framework.processing.map_processor import MapProcessor
from deep_segmentation_framework.resources import *


# Import the code for the DockWidget
from .deep_segmentation_framework_dockwidget import DeepSegmentationFrameworkDockWidget, InferenceInput
import os.path


class DeepSegmentationFramework:
    """QGIS Plugin Implementation."""

    def __init__(self, iface: QgisInterface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        # Save reference to the QGIS interface
        self.iface = iface

        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)

        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'DeepSegmentationFramework_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            QCoreApplication.installTranslator(self.translator)

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&Deep Segmentation Framework')
        # TODO: We are going to let the user set this up in a future iteration
        self.toolbar = self.iface.addToolBar(u'DeepSegmentationFramework')
        self.toolbar.setObjectName(u'DeepSegmentationFramework')

        #print "** INITIALIZING DeepSegmentationFramework"

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
        return QCoreApplication.translate('DeepSegmentationFramework', message)


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

        icon_path = ':/plugins/deep_segmentation_framework/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'Deep Segmentation Framework'),
            callback=self.run,
            parent=self.iface.mainWindow())

    #--------------------------------------------------------------------------

    def onClosePlugin(self):
        """Cleanup necessary items here when plugin dockwidget is closed"""

        #print "** CLOSING DeepSegmentationFramework"

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

        #print "** UNLOAD DeepSegmentationFramework"

        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&Deep Segmentation Framework'),
                action)
            self.iface.removeToolBarIcon(action)
        # remove the toolbar
        del self.toolbar

    #--------------------------------------------------------------------------

    def _layers_changed(self, _):
        self.dockwidget.update_input_layer_selection(QgsProject.instance().mapLayers())

    def run(self):
        """Run method that loads and starts the plugin"""

        if not self.pluginIsActive:
            self.pluginIsActive = True

            #print "** STARTING DeepSegmentationFramework"

            # dockwidget may not exist if:
            #    first run of plugin
            #    removed on close (see self.onClosePlugin method)
            if self.dockwidget is None:
                # Create the dockwidget (after translation) and keep reference
                self.dockwidget = DeepSegmentationFrameworkDockWidget()
                self._layers_changed(None)
                QgsProject.instance().layersAdded.connect(self._layers_changed)
                QgsProject.instance().layersRemoved.connect(self._layers_changed)

            # connect to provide cleanup on closing of dockwidget
            self.dockwidget.closingPlugin.connect(self.onClosePlugin)
            self.dockwidget.do_something_signal.connect(self._do_something)
            self.dockwidget.run_inference_signal.connect(self._run_inference)

            self.iface.addDockWidget(Qt.RightDockWidgetArea, self.dockwidget)
            self.dockwidget.show()

    def round_extent(self, extent, units_per_pixel_xy: tuple):
        # rounded to layer "grid" for pixels. Grid starts at rlayer_extent.xMinimum
        # with resolution of rlayer_units_per_pixel
        x_min = int(extent.xMinimum() / units_per_pixel_xy[0]) * units_per_pixel_xy[0]
        x_max = int(extent.xMaximum() / units_per_pixel_xy[0]) * units_per_pixel_xy[0]
        y_min = int(extent.yMinimum() / units_per_pixel_xy[1]) * units_per_pixel_xy[1]
        y_max = int(extent.yMaximum() / units_per_pixel_xy[1]) * units_per_pixel_xy[1]
        extent.setXMinimum(x_min)
        extent.setXMaximum(x_max)
        extent.setYMinimum(y_min)
        extent.setYMaximum(y_max)
        return extent

    def _get_extent_for_processing(self, rlayer, use_entire_field: bool, mask_layer_name: str = None):
        """
        :param use_entire_field: Whether extent for the entire field should be given.
        Otherwise, only active map area will be used
        """
        if use_entire_field:
            active_extent = rlayer.extent()
        elif mask_layer_name is not None:
            active_extent = rlayer.extent()
            active_extent = QgsProject.instance().mapLayersByName(mask_layer_name)[0].getGeometry(0)
            active_extent.convertToSingleType()
            active_extent = active_extent.boundingBox()
        else:
            # transform visible extent from mapCanvas CRS to layer CRS
            active_extent_canvas_crs = self.iface.mapCanvas().extent()
            canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
            t = QgsCoordinateTransform()
            t.setSourceCrs(canvas_crs)
            t.setDestinationCrs(rlayer.crs())
            active_extent = t.transform(active_extent_canvas_crs)

        return active_extent

    def _do_something(self):
        # After pressing 'do_something' button
        print('Doing something...')

        # proj = QgsProject.instance()
        # value, _ = proj.readNumEntry(PLUGIN_NAME, 'testcounter', 0)
        # value += 1
        # proj.writeEntry(PLUGIN_NAME, 'testcounter', value)
        # self._show_active_raster_as_image()

        # self.iface.messageBar().pushMessage("Info", "hello", level=Qgis.Critical)
        # self.iface.messageBar().pushMessage("Info", f"hello {value}", level=Qgis.Success)
        # task = TestTask('my task', 10)
        # QgsApplication.taskManager().addTask(task)
        # QgsMessageLog.logMessage("doing something...", LOG_TAB_NAME, level=Qgis.Info)
        # print(f'{value = }')

    def _get_active_extent_rounded(self, rlayer, inference_parameters: InferenceParameters) -> QgsRectangle:
        rlayer_extent = rlayer.extent()
        rlayer_units_per_pixel = rlayer.rasterUnitsPerPixelX(), rlayer.rasterUnitsPerPixelY()

        active_extent = self._get_extent_for_processing(rlayer=rlayer,
                                                        use_entire_field=inference_parameters.entire_field,
                                                        mask_layer_name=inference_parameters.layer_name)

        active_extent_intersect = active_extent.intersect(rlayer_extent)
        active_extent_rounded = self.round_extent(active_extent_intersect, rlayer_units_per_pixel)
        return active_extent_rounded

    def _run_inference(self, inference_input : InferenceInput):
        inference_parameters = inference_input.inference_parameters

        if self._map_processor and self._map_processor.is_busy():
            msg = "Error! Processing already in progress! Please wait or cancel previous task."
            self.iface.messageBar().pushMessage(PLUGIN_NAME, msg, level=Qgis.Critical)
            return

        rlayer = QgsProject.instance().mapLayers()[inference_input.input_layer_id]
        if rlayer is None:
            msg = "Error! Please select the layer to process first!"
            self.iface.messageBar().pushMessage(PLUGIN_NAME, msg, level=Qgis.Critical)
            return

        if isinstance(rlayer, QgsVectorLayer):
            msg = "Error! Please select a raster layer (vector layer selected)"
            self.iface.messageBar().pushMessage(PLUGIN_NAME, msg, level=Qgis.Critical)
            return

        processed_extent = self._get_active_extent_rounded(rlayer, inference_parameters)

        self._map_processor = MapProcessor(rlayer=rlayer,
                                           processed_extent=processed_extent,
                                           inference_parameters=inference_parameters)
        self._map_processor.finished_signal.connect(self._map_processor_finished)
        self._map_processor.show_img_signal.connect(self._show_img)
        QgsApplication.taskManager().addTask(self._map_processor)

    @staticmethod
    def _show_img(img, window_name):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 800)
        cv2.imshow(window_name, img)
        cv2.waitKey(1)

    def _map_processor_finished(self, error_msg):
        print(f'_map_processor_finished. {error_msg = }')
        if error_msg:
            msg = f'Error! Processing error: "{error_msg}"!'
            self.iface.messageBar().pushMessage(PLUGIN_NAME, msg, level=Qgis.Critical)
        else:
            msg = 'Processing finished!'
            self.iface.messageBar().pushMessage(PLUGIN_NAME, msg, level=Qgis.Success)
        self._map_processor = None


"""
Writing function of the entire image (from Raster Vision plugin).
But it behaves strangle for e.g. google earth data, and we don't want to create a temporary file for entire map
(for unlimited map we would need an limit anyway)

from qgis.core import (QgsProject,
                       QgsRasterFileWriter,
                       QgsRasterLayer,
                       QgsRasterPipe)

def export_raster_layer(layer, path):
    provider = layer.dataProvider()
    renderer = layer.renderer()
    pipe = QgsRasterPipe()
    pipe.set(provider.clone())
    pipe.set(renderer.clone())
    
    file_writer = QgsRasterFileWriter(path)
    file_writer.writeRaster(
        pipe,
        provider.xSize(),
        provider.ySize(),
        provider.extent(),
        provider.crs())        
        
with TemporaryDirectory(dir=settings.get_working_dir()) as tmp_dir:
    path = os.path.join(tmp_dir, "{}.tif".format(layer_name))
"""


"""
Logging:
self.iface.messageBar().pushMessage("Info", "hello", level=Qgis.Success)
QgsMessageLog.logMessage("Widget setup", LOG_TAB_NAME, level=Qgis.Info)


QgsProject.instance().mapLayers()
qgis.utils.iface.activeLayer()


Raster Layers:
rlayer = qgis.utils.iface.activeLayer() # some raster layer
rlayer.width(), rlayer.height()  # dimension of the layer (in px)
rlayer.extent()  # get the extent of the layer as QgsRectangle

rlayer.rasterType()  # value 2 is multiband
rlayer.bandCount()
print(rlayer.bandName(1))

val, res = rlayer.dataProvider().sample(QgsPointXY(638907, 5802464), 1)  # get data point  [ sample (const QgsPointXY &point, int band, bool *ok=nullptr, const QgsRectangle &boundingBox=QgsRectangle(), int width=0, int height=0, int dpi=96) ]
rlayer.dataProvider().identify(QgsPointXY(638907, 5802464), QgsRaster.IdentifyFormatValue).results()
rlayer.rasterUnitsPerPixelX()


# seting rendering mode 
fcn = QgsColorRampShader()
fcn.setColorRampType(QgsColorRampShader.Discrete)
lst = [ QgsColorRampShader.ColorRampItem(0, QColor(0,255,0)),
      QgsColorRampShader.ColorRampItem(255, QColor(255,255,0)) ]
fcn.setColorRampItemList(lst)
shader = QgsRasterShader()
shader.setRasterShaderFunction(fcn)
renderer = QgsSingleBandPseudoColorRenderer(rlayer.dataProvider(), 1, shader)
rlayer.setRenderer(renderer)


# Editing raster data
block = QgsRasterBlock(Qgis.Byte, 2, 2)
block.setData(b'\xaa\xbb\xcc\xdd')
provider = rlayer.dataProvider()
provider.setEditable(True)
provider.writeBlock(block, 1, 0, 0)
provider.setEditable(False)


# Project settings
proj = QgsProject.instance()
value, _ = proj.readNumEntry(PLUGIN_NAME, 'testcounter', 0)
value += 1
proj.writeEntry(PLUGIN_NAME, 'testcounter', value)
        

# Running background task:

# e.g. after clicking the button:
task = TestTask('my task', 10)
QgsApplication.taskManager().addTask(task)



# Access plugin in qgis console
my_plugin = qgis.utils.plugins['My Plugin']

"""
