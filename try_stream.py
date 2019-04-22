from matplotlib import pyplot as plt

from data import ImageStream, AugmentationConfig


dsroot = "/data/Datasets/SCUT-FBP5500_v2/"
aug = AugmentationConfig()
# aug.rotation_degree_range = None
# aug.centroid_max_perturbance = None
# aug.scale_ratio_range = None
# aug.flip_probability = None

stream = ImageStream(dsroot + "Files.npy", dsroot + "ratings.npy", dsroot + "Images/",
                     batch_size=1,
                     augmentation_config=aug)

for [image], [rating] in stream:
    print(stream.processor.augmentation_params.__dict__)
    plt.imshow(image[..., ::-1])
    plt.title("Score: {}".format(rating))
    plt.show()
