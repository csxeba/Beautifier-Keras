import yaml
import pandas as pd

from model.fcn import FCN
from data.loader import ImageStream
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger

config = yaml.load(open("config.yml"))
root = config["dataset_root"]
table = pd.read_excel(root + "All_Ratings.xlsx")
stream = ImageStream(table["Filename"], table["Rating"], root + "Images/")

fcn = FCN()
fcn.build_beaty_predictor(input_shape=(240, 240, 3))

fcn.training_aspect.fit_generator(
    stream, stream.steps_per_epoch, epochs=30,
    callbacks=[
        TensorBoard("logs/"),
        ModelCheckpoint("beautifier_predictor_e{epoch}.h5", period=5, save_weights_only=True),
        ModelCheckpoint("beautifier_predictor_latest.h5", save_weights_only=True),
        ModelCheckpoint("beautifier_predictor_best.h5", save_best_only=True, save_weights_only=True),
        CSVLogger("train_log.csv")
    ]
)
fcn.testing_aspect.save_weights(root + "beautifier_predictor_final.h5")
