# -*- coding: utf-8 -*-

import numpy as np


class ImageUtils(object):

    @staticmethod
    def fillBlackBorderWithRandomNoise(image=np.ndarray((0, 0, 0)), mean=193):
        # Create a mask from the actual first-frame dimensions.
        # The previous implementation incorrectly assumed width == height by
        # creating a (image.shape[0], image.shape[0]) mask. That fails for
        # valid rectangular inputs such as 512 x 507.
        mask = np.logical_not(image[:, :, 0].astype(bool))

        # Fill the black border with the configured mean value.
        noise_array = np.full(image[mask].shape, mean, np.uint8)
        image[mask] = noise_array
        return image
