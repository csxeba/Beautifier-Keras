import numpy as np
import cv2

from data.augmentation import AugmentationParams


class ImageProcessor:

    EYE = np.eye(3)

    def __init__(self, network_input_shape=(240, 240), augmentation_config=None):
        self.augmentation_params = AugmentationParams()
        self.augment = bool(augmentation_config)
        self.augmentation_config = augmentation_config
        self.input_shape = network_input_shape

    def _get_rotation(self, degrees, scale):
        canvas = self.EYE.copy()
        rad = np.deg2rad(degrees)
        sdeg = np.sin(rad) * scale
        cdeg = np.cos(rad) * scale
        canvas[0, 0] = canvas[1, 1] = cdeg
        canvas[0, 1] = sdeg
        canvas[1, 0] = -sdeg
        return canvas

    def _get_translation(self, x, y):
        # canvas = np.array([[1., 0., x],
        #                    [0., 1., y],
        #                    [0., 0., 1.]])
        canvas = self.EYE.copy()
        canvas[0, 2] = x
        canvas[1, 2] = y
        return canvas

    def _get_scale(self, scale):
        canvas = self.EYE.copy()
        canvas[0, 0] = canvas[1, 1] = scale
        return canvas

    def _get_flip(self, flip):
        canvas = self.EYE.copy()
        canvas[0, 0] = flip
        return canvas

    def get_transformation(self, centroid):
        center_to_origin = self._get_translation(*(-centroid))
        scale = self._get_scale(self.augmentation_params.scale_ratio)
        rotation = self._get_rotation(self.augmentation_params.rotation_degree, self.augmentation_params.scale_ratio)
        flip = self._get_flip(self.augmentation_params.flip_value)
        new_center = (np.array(self.input_shape) + self.augmentation_params.centroid_perturbance) // 2
        center_to_center = self._get_translation(*new_center)

        transformation = center_to_center @ flip @ scale @ rotation @ center_to_origin

        return transformation[:2]

    def __call__(self, image):
        centroid = np.array(image.shape[:2]) // 2
        if self.augment:
            self.augmentation_params.randomize(self.augmentation_config)
            transformation = self.get_transformation(centroid)
            data = cv2.warpAffine(image,
                                  transformation,
                                  self.input_shape,
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(127, 127, 127))
        else:
            cx, cy = centroid
            data = image[cx-120:cx+120, cy-120:cy+120]

        return data
