General Information for Developers
==================================

.. note::

  Project github: https://github.com/PUTvision/qgis-plugin-deepness

  Project issues tracker: https://github.com/PUTvision/qgis-plugin-deepness/issues



===================================
Development environemnt preparation
===================================

Please visit README.md file in the main repository directory for instructions how to prepare the development environment.



========================
Overview of core modules
========================

Below a general overview of core modules and classes is presented:
 * :code:`Deepness` - Plugin entry class, managing UI and connecting it with processing functionality
 * :code:`DeepnessDockWidget` - Core UI widget, which includes further smaller widgets. Actions started in UI (button clicks) are handled here
 * :code:`MapProcessingParameters` - stores parameters from the UI, which will be passed to the further processing module
(with different child classes for specialized models, e.g. :code:`SegmentationParameters`)
 * :code:`MapProcessor` - base class for processing the ortophoto. Actual processing done in specialized class, e.g. :code:`MapProcessorSegmentation`. :code:`MapProcessor` uses a :code:`ModelBase` as the core processor of single tiles.
Objects of this class are created and managed by the :code:`Deepness`.
 * :code:`ModelBase` - Wraps the ONNX model used during processing into a common interface (with different child classes for specialized models, e.g. :code:`Segmentor` for segmentation)

Apart from these core modules, there are many utility files, but not required to understand the core logic.


====================
Processing data flow
====================
Core data and logic flow, on example of segmentation model on a ortophoto:
 * user selects some parameters, Segmentation model and ortophoto layer to process and then presses *Run* button
 * :code:`DeepnessDockWidget` is triggered, parameters from UI forms are written to :code:`SegmentationParameters` object
 * :code:`DeepnessDockWidget` triggers :code:`Deepness` and passes the processing parameters
 * :code:`Deepness` creates and starts :code:`MapProcessorSegmentation` object, initiating the processing with :code:`SegmentationParameters`
 * :code:`MapProcessorSegmentation` runs in a separate processing thread, iterating over the ortophoto tile by tile. Once the processing is done, result layer is created and 'Deepness' is triggered
 * :code:`Deepness` displays processing result (success or error message)
