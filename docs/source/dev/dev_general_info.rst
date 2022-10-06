General Information for Developers
=====

Project github: https://github.com/PUTvision/qgis-deep-segmentation-framework

Project issues tracker: https://github.com/PUTvision/qgis-deep-segmentation-framework/issues

Please visit README.md file in the main repository directory for instructions how to prepare the development environment.



Below a general overview of core modules and classes is presented:
- 'DeepSegmentationFramework' - Plugin entry class, managing UI and connecting it with processing functionality
- 'DeepSegmentationFrameworkDockWidget' - Core UI widget, which includes further smaller widgets. Actions started in UI (button clicks) are handled here
- 'MapProcessingParameters' - stores parameters from the UI, which will be passed to the further processing module
(with different child classes for specialized models, e.g. 'SegmentationParameters')
- 'MapProcessor' - base class for processing the ortophoto. Actual processing done in specialized class,
e.g. 'MapProcessorSegmentation'. 'MapProcessor' uses a 'ModelBase' as the core processor of single tiles.
Objects of this class are created and managed by the 'DeepSegmentationFramework'.
- 'ModelBase' - Wraps the ONNX model used during processing into a common interface
(with different child classes for specialized models, e.g. 'Segmentor' for segmentation)

Apart from these core modules, there are many utility files, but not required to understand the core logic.


Core data and logic flow, on example of segmentation model on a ortophoto:
- user selects some parameters, Segmentation model and ortophoto layer to process and then presses 'Run' button
- 'DeepSegmentationFrameworkDockWidget' is triggered, parameters from UI forms are written to 'SegmentationParameters' object
- 'DeepSegmentationFrameworkDockWidget' triggers 'DeepSegmentationFramework' and passes the processing parameters
- 'DeepSegmentationFramework' creates and starts 'MapProcessorSegmentation' object, initiating the processing with 'SegmentationParameters'
- 'MapProcessorSegmentation' runs in a separate processing thread, iterating over the ortophoto tile by tile. Once the processing is done, result layer is created and 'DeepSegmentationFramework' is triggered
- 'DeepSegmentationFramework' displays processing result (success or error message)
