""" This file defines possible outcomes of map processing
"""


class MapProcessingResult:
    """
    Base class for signaling finished processing result
    """

    def __init__(self, message: str):
        self.message = message


class MapProcessingResultSuccess(MapProcessingResult):
    """
    Processing result on success
    """

    def __init__(self, message: str = ''):
        super().__init__(message)


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
