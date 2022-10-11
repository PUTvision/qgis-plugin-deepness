Installation
============


============
Requirements
============

* Ubuntu
  
  * (option 1) Install requirements using system Python interpreter:
  
  .. code-block:: 

    python3 -m pip install opencv-python-headless onnxruntime-gpu

  * (option 2) Run QGIS and Python Console. Then call command:

  .. code-block:: 

    import pip; pip.main(['install', 'opencv-python-headless', 'onnxruntime-gpu'])


* Windows
  
  * Go to QGIS installation path (for example :code:`C:\Program Files\QGIS 3.26.3\`)
  
  * Run :code:`OSGeo4W.bat` and type installation command:
  
  .. code-block:: 

    python3 -m pip install opencv-python-headless onnxruntime-gpu

* MacOS - SOON

===================
Plugin installation
===================

* (option 1) Install using QGIS plugin browser
  
  .. note:: 

    The repository has not yet been pushed out to the QGIS plugins collection.

  * Run QGIS
  
  * Open: Plugins->Manage and Install Plugins
  
  * Select: Not installed

  * Type in the "Search..." field plugin name

  * Select from list and click the Install button


* (option 2) Install using downloaded ZIP

  * Go to the plugin repository: `https://github.com/PUTvision/qgis-deep-segmentation-framework <https://github.com/PUTvision/qgis-deep-segmentation-framework>`_

  * From the right panel, select Release and download the latest version

  * Run QGIS

  * Open: Plugins->Manage and Install Plugins
  
  * Select: Install from ZIP

  * Select the ZIP file using system prompt

  * Click the Install Plugin button
