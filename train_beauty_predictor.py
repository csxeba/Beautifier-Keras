from model import vgg as model
from data import ImageStream, AugmentationConfig
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger

root = "data/SCUT/"
aug = AugmentationConfig()
stream = ImageStream.from_cache(root + "Images/", batch_size=32, augmentation_config=aug)

model.fit_generator(
    stream, stream.steps_per_epoch,
    epochs=30,
    callbacks=[
        TensorBoard("logs/"),
        ModelCheckpoint("beautifier_predictor_e{epoch}.h5", period=5, save_weights_only=True),
        ModelCheckpoint("beautifier_predictor_latest.h5", save_weights_only=True),
        CSVLogger("train_log.csv")
    ]
)
