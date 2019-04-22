import numpy as np


class AugmentationConfig:

    def __init__(self,
                 rotation_degree_range=(-45, 45),
                 centroid_max_perturbance=40,
                 scale_ratio_range=(0.6, 1.1),
                 flip_probability=0.5):

        self.rotation_degree_range = rotation_degree_range
        self.centroid_max_perturbance = centroid_max_perturbance
        self.scale_ratio_range = scale_ratio_range
        self.flip_probability = flip_probability


class AugmentationParams:

    def __init__(self):
        self.rotation_degree = None
        self.centroid_perturbance = None
        self.scale_ratio = None
        self.do_flip = None
        self.reset()

    def randomize(self, config: AugmentationConfig):
        self.reset()
        if config.rotation_degree_range is not None:
            self.rotation_degree = np.random.randint(*config.rotation_degree_range)
        if config.centroid_max_perturbance is not None:
            centroid_perturb = -config.centroid_max_perturbance, config.centroid_max_perturbance
            self.centroid_perturbance = np.random.randint(*centroid_perturb, size=2)
        if config.scale_ratio_range is not None:
            self.scale_ratio = np.random.uniform(*config.scale_ratio_range)
        if config.flip_probability is not None:
            self.do_flip = np.random.random() < config.flip_probability

    def reset(self):
        self.rotation_degree = 0
        self.centroid_perturbance = np.zeros(2)
        self.scale_ratio = 1.
        self.do_flip = False

    @property
    def flip_value(self):
        return -1 if self.do_flip else 1
