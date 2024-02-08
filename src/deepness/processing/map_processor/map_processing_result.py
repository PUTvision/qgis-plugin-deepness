""" This file defines possible outcomes of map processing
"""


from typing import Callable


class MapProcessingResult:
    """
    Base class for signaling finished processing result
    """

    def __init__(self, message: str, gui_delegate: Callable | None = None):
        """
        :param message: message to be shown to the user
        :param gui_delegate: function to be called in GUI thread, as it is not safe to call GUI functions from other threads
        """
        self.message = message
        self.gui_delegate = gui_delegate


class MapProcessingResultSuccess(MapProcessingResult):
    """
    Processing result on success
    """

    def __init__(self, message: str = '', gui_delegate: Callable | None = None):
        super().__init__(message=message, gui_delegate=gui_delegate)


class MapProcessingResultFailed(MapProcessingResult):
    """
    Processing result on error
    """

    def __init__(self, error_message: str, exception=None):
        super().__init__(error_message)
        self.exception = exception


class MapProcessingResultCanceled(MapProcessingResult):
    """
    Processing when processing was aborted
    """

    def __init__(self):
        super().__init__(message='')
