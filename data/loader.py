import os

import numpy as np
import cv2

from .processor import ImageProcessor
from .augmentation import AugmentationConfig


class ImageStream:

    def __init__(self,
                 filenames_cache,
                 rates_cache,
                 image_root,
                 batch_size=32,
                 augmentation_config: AugmentationConfig=None):

        self.files = np.array([os.path.join(image_root, f) for f in np.load(filenames_cache)])
        self.rates = np.load(rates_cache) / 5.
        self.batch_size = batch_size
        self.indices = np.arange(len(self.files))
        self.processor = ImageProcessor(network_input_shape=(240, 240),
                                        augmentation_config=augmentation_config)
        self._iterator = self.make_iterator()

    @property
    def steps_per_epoch(self):
        return int(round(len(self.files) / self.batch_size))

    def load_image(self, path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("No such image file:", path)
        return image

    def make_iterator(self):
        while 1:
            X = np.empty((self.batch_size, 240, 240, 3), dtype="uint8")
            Y = np.empty((self.batch_size,), dtype="float32")
            for i, index in enumerate(np.random.choice(self.indices, size=self.batch_size)):
                image = self.load_image(self.files[index])
                X[i] = self.processor(image)
                Y[i] = self.rates[index]
            yield X, Y

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator)
