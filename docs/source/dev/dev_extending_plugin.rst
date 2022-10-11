Extending plugin functionality
==============================

Plugin is easily extendable for new types of parameters, models and outputs.

Below, a use case of adding support for a new regression model type (with new parameters in UI) is presented:
 * Edit the main UI form (or appropriate child widget), e.g. :code:`deep_segmentation_framework_dockwidget_base.ui`.
    Add new widgets as needed.
 * Add parsing of these UI values to :code:`RegressionParameters` within :code:`deep_segmentation_framework_dockwidget_base.py` file
 * Add routines for saving and loading these parameters in project settings, as it is done for other parameters (file :code:`deep_segmentation_framework_dockwidget_base.py`)
 * Create a new model type in directory :code:`processing/models` (or derive from an existing one, e.g. :code:`Regressor`, and overwrite required functionality)
 * Register created model type within :code:`ModelType` class
 * Add a unit test in :code:`test` directory. Use this test to validate and debug the logic before running in UI
