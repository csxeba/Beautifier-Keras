import os

import numpy as np
import cv2

from .processor import ImageProcessor
from .augmentation import AugmentationConfig


class ImageStream:

    def __init__(self,
                 filenames,
                 ratings,
                 image_root,
                 batch_size=32,
                 augmentation_config: AugmentationConfig=None):

        self.files = np.array(filenames)
        self.image_root = image_root
        self.ratings = ratings
        self.batch_size = batch_size
        self.indices = np.arange(len(self.files))
        self.processor = ImageProcessor(network_input_shape=(240, 240),
                                        augmentation_config=augmentation_config)
        self._iterator = self.make_iterator()

    @classmethod
    def from_pandas(cls, root, batch_size=32, augmentation_config: AugmentationConfig=None):

        import pandas as pd
        image_root = os.path.join(root, "Images/")
        table = pd.read_excel(root + "All_Ratings.xlsx")
        files = table["Filename"].as_matrix()
        rates = table["Rating"].as_matrix()
        return cls(files, rates, image_root, batch_size, augmentation_config)

    @classmethod
    def from_arrays(cls, root, batch_size=32, augmentation_config: AugmentationConfig=None):
        image_root = os.path.join(root, "Images/")
        npz = np.load(os.path.join(root, "beauty_cache.npz"))
        return cls(npz["files"], npz["rates"], image_root, batch_size, augmentation_config)

    @classmethod
    def from_cache(cls, root, batch_size=32, augmentation_config: AugmentationConfig=None):
        if os.path.exists(os.path.join(root, "beauty_cache.npz")):
            return cls.from_arrays(root, batch_size, augmentation_config)

        print(["ImageStream: No cache, caching now..."])
        obj = cls.from_pandas(root, batch_size, augmentation_config)
        np.savez(os.path.join(root, "beauty_cache.npz"), files=obj.files, rates=obj.ratings)
        return obj

    @property
    def steps_per_epoch(self):
        return int(round(len(self.files) / self.batch_size))

    def load_image(self, path):
        image = cv2.imread(os.path.join(self.image_root, path), cv2.IMREAD_COLOR)
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
                Y[i] = self.ratings[index]
            yield X / 255. - .5, Y / 5. - .5

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator)
