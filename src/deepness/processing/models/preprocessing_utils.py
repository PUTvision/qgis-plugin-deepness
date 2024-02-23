import numpy as np

from deepness.common.processing_parameters.standardization_parameters import StandardizationParameters


def limit_channels_number(tiles_batched: np.array, limit: int) -> np.array:
    """ Limit the number of channels in the input image to the model

    :param tiles_batched: Batch of tiles
    :param limit: Number of channels to keep
    :return: Batch of tiles with limited number of channels
    """
    return tiles_batched[:, :, :, :limit]


def normalize_values_to_01(tiles_batched: np.array) -> np.array:
    """ Normalize the values of the input image to the model to the range [0, 1]

    :param tiles_batched: Batch of tiles
    :return: Batch of tiles with values in the range [0, 1], in float32
    """
    return np.float32(tiles_batched * 1./255.)


def standardize_values(tiles_batched: np.array, params: StandardizationParameters) -> np.array:
    """ Standardize the input image to the model

    :param tiles_batched: Batch of tiles
    :param params: Parameters for standardization of type STANDARIZE_PARAMS
    :return: Batch of tiles with standardized values
    """
    return (tiles_batched - params.mean) / params.std


def transpose_nhwc_to_nchw(tiles_batched: np.array) -> np.array:
    """ Transpose the input image from NHWC to NCHW

    :param tiles_batched: Batch of tiles in NHWC format
    :return: Batch of tiles in NCHW format
    """
    return np.transpose(tiles_batched, (0, 3, 1, 2))
