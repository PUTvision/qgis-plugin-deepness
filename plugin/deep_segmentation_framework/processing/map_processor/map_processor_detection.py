from typing import List

from deep_segmentation_framework.common.processing_parameters.detection_parameters import DetectionParameters
from deep_segmentation_framework.common.defines import IS_DEBUG
from deep_segmentation_framework.processing import processing_utils
from deep_segmentation_framework.processing.map_processor.map_processor import MapProcessor
from deep_segmentation_framework.processing.tile_params import TileParams

if IS_DEBUG:
    pass


class MapProcessorDetection(MapProcessor):
    """
    Process the entire map for the detection models, which produce bounding boxes
    """

    def __init__(self,
                 params: DetectionParameters,
                 **kwargs):
        super().__init__(
            params=params,
            **kwargs)
        self.detection_parameters = params
        self.model = params.model

    def _run(self):
        all_bounding_boxes = []  # type: List[...]
        for tile_img, tile_params in self.tiles_generator():
            if self.isCanceled():
                return False

            bounding_boxes_in_tile = self._process_tile(tile_img, tile_params)
            all_bounding_boxes += bounding_boxes_in_tile

        all_bounding_boxes_suppressed = self.apply_non_maximum_supression(all_bounding_boxes)

        all_bounding_boxes_restricted = self.limit_bounding_boxes_to_processed_area(all_bounding_boxes_suppressed)

        self._create_vlayer_from_mask_for_base_extent(all_bounding_boxes_restricted)

        return True

    def limit_bounding_boxes_to_processed_area(self, bounding_boxes):
        """
        Limit all bounding boxes to the constrained area that we process.
        E.g. if we are detecting peoples in a circle, we don't want to count peoples in the entire rectangle
        :return:
        """
        self.area_mask_img = processing_utils.create_area_mask_image(
            vlayer_mask=self.vlayer_mask,
            extended_extent=self.extended_extent,
            rlayer_units_per_pixel=self.rlayer_units_per_pixel,
            image_shape_yx=[self.img_size_y_pixels, self.img_size_x_pixels])

        # if bounding box is not in the area_mask_img (at least in some percentage) - remove it
        return bounding_boxes

    def _create_vlayer_for_output_bounding_boxes(self, bounding_boxes):
        # TODO - create group with a layer for each output
        pass

    def apply_non_maximum_supression(self, bounding_boxes):
        return bounding_boxes

    def convert_bounding_boxes_to_absolute_positions(self, bounding_boxes_relative, tile_params: TileParams):
        # TODO - implement
        return bounding_boxes_relative

    def _process_tile(self, tile_img: np.ndarray, tile_params: TileParams) -> np.ndarray:
        # TODO - create proper mapping for output channels
        bounding_boxes_relative = self.model.process(tile_img)

        bounding_boxes_absolute_positions = self.convert_bounding_boxes_to_absolute_positions(
            bounding_boxes_relative, tile_params)

        return bounding_boxes_absolute_positions

