from typing import Optional

import cv2
import numpy as np
from qgis.core import QgsRectangle

from deepness.common.processing_parameters.map_processing_parameters import MapProcessingParameters


class TileParams:
    def __init__(self,
                 x_bin_number,
                 y_bin_number,
                 x_bins_number,
                 y_bins_number,
                 params: MapProcessingParameters,
                 rlayer_units_per_pixel,
                 processing_extent):
        self.x_bin_number = x_bin_number
        self.y_bin_number = y_bin_number
        self.x_bins_number = x_bins_number
        self.y_bins_number = y_bins_number
        self.stride_px = params.processing_stride_px
        self.start_pixel_x = x_bin_number * self.stride_px
        self.start_pixel_y = y_bin_number * self.stride_px
        self.params = params
        self.rlayer_units_per_pixel = rlayer_units_per_pixel

        self.extent = self._calculate_extent(processing_extent)  # type: QgsRectangle  # tile extent in CRS cordinates

    def _calculate_extent(self, processing_extent):
        tile_extent = QgsRectangle(processing_extent)  # copy
        x_min = processing_extent.xMinimum() + self.start_pixel_x * self.rlayer_units_per_pixel
        y_max = processing_extent.yMaximum() - self.start_pixel_y * self.rlayer_units_per_pixel
        tile_extent.setXMinimum(x_min)
        # extent needs to be on the further edge (so including the corner pixel, hence we do not subtract 1)
        tile_extent.setXMaximum(x_min + self.params.tile_size_px * self.rlayer_units_per_pixel)
        tile_extent.setYMaximum(y_max)
        y_min = y_max - self.params.tile_size_px * self.rlayer_units_per_pixel
        tile_extent.setYMinimum(y_min)
        return tile_extent

    def get_slice_on_full_image_for_entire_tile(self):
        """
        Slice to get the entire tile from "full final image,
        including the overlapping parts.
        :return Slice to be used on the full image
        """

        # 'core' part of the tile (not overlapping with other tiles), for sure copied for each tile
        x_min = self.start_pixel_x
        x_max = self.start_pixel_x + self.params.tile_size_px - 1
        y_min = self.start_pixel_y
        y_max = self.start_pixel_y + self.params.tile_size_px - 1

        roi_slice = np.s_[y_min:y_max + 1, x_min:x_max + 1]
        return roi_slice

    def get_slice_on_full_image_for_copying(self):
        """
        As we are doing processing with overlap, we are not going to copy the entire tile result to final image,
        but only the part that is not overlapping with the neighbouring tiles.
        Edge tiles have special handling too.

        :return Slice to be used on the full image
        """
        half_overlap = (self.params.tile_size_px - self.stride_px) // 2

        # 'core' part of the tile (not overlapping with other tiles), for sure copied for each tile
        x_min = self.start_pixel_x + half_overlap
        x_max = self.start_pixel_x + self.params.tile_size_px - half_overlap - 1
        y_min = self.start_pixel_y + half_overlap
        y_max = self.start_pixel_y + self.params.tile_size_px - half_overlap - 1

        # edge tiles handling
        if self.x_bin_number == 0:
            x_min -= half_overlap
        if self.y_bin_number == 0:
            y_min -= half_overlap
        if self.x_bin_number == self.x_bins_number-1:
            x_max += half_overlap
        if self.y_bin_number == self.y_bins_number-1:
            y_max += half_overlap

        roi_slice = np.s_[y_min:y_max + 1, x_min:x_max + 1]
        return roi_slice

    def get_slice_on_tile_image_for_copying(self, roi_slice_on_full_image = None):
        """
        Similar to _get_slice_on_full_image_for_copying, but ROI is a slice on the tile
        """
        if not roi_slice_on_full_image:
            roi_slice_on_full_image = self.get_slice_on_full_image_for_copying()

        r = roi_slice_on_full_image
        roi_slice_on_tile = np.s_[
                            r[0].start - self.start_pixel_y:r[0].stop - self.start_pixel_y,
                            r[1].start - self.start_pixel_x:r[1].stop - self.start_pixel_x
                            ]
        return roi_slice_on_tile

    def is_tile_within_mask(self, mask_img: Optional[np.ndarray]):
        """
        To check if tile
        :param mask_img:
        :return:
        """
        if mask_img is None:
            return True  # if we don't have a mask, we are going to process all tiles

        roi_slice = self.get_slice_on_full_image_for_copying()
        mask_roi = mask_img[roi_slice]
        # check corners first
        if mask_roi[0, 0] and mask_roi[1, -1] and mask_roi[-1, 0] and mask_roi[-1, -1]:
            return True  # all corners in mask, almost for sure a good tile

        coverage_percentage = cv2.countNonZero(mask_roi) / (mask_roi.shape[0] * mask_roi.shape[1]) * 100
        return coverage_percentage > 0  # TODO - for training we can use tiles with higher coverage only

    def set_mask_on_full_img(self, full_result_img, tile_result):
        roi_slice_on_full_image = self.get_slice_on_full_image_for_copying()
        roi_slice_on_tile_image = self.get_slice_on_tile_image_for_copying(roi_slice_on_full_image)
        full_result_img[roi_slice_on_full_image] = tile_result[roi_slice_on_tile_image]

    def get_entire_tile_from_full_img(self, full_result_img) -> np.ndarray:
        roi_slice_on_full_image = self.get_slice_on_full_image_for_entire_tile()
        img = full_result_img[roi_slice_on_full_image]
        return img
