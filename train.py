from scipy.io import loadmat

from pathlib import Path
import numpy as np
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import SGD
from keras.utils import np_utils
from wide_resnet import WideResNet


def load_data(target_mat_path):
    data = loadmat(target_mat_path)

    return data["image"], data["gender"][0], data["age"][0], data["db"][0], data["img_size"][0, 0], data["min_score"][0, 0]


class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.2
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.04
        return self.initial_lr * 0.008


def main():
    cleaned_mat_path = "data/wiki.mat"
    batch_size = 64
    epochs_num = 25
    lr = 0.1
    depth = 16
    k = 8
    validation_split = 0.1
    model_path = Path(__file__).resolve().parent.joinpath("model")
    model_path.mkdir(parents=True, exist_ok=True)

    # split data into train set and test set with the ratio 9:1
    image, gender, age, _, image_size, _ = load_data(cleaned_mat_path)
    data_img_x = image
    data_gender_y = np_utils.to_categorical(gender, 2)
    data_age_y = np_utils.to_categorical(age, 101)
    data_num = len(data_img_x)
    indexes = np.arange(data_num)
    np.random.shuffle(indexes)
    data_img_x = data_img_x[indexes]
    data_gender_y = data_gender_y[indexes]
    data_age_y = data_age_y[indexes]
    train_num = int(data_num * (1 - validation_split))
    train_x = data_img_x[:train_num]
    test_x = data_img_x[train_num:]
    train_gender_y = data_gender_y[:train_num]
    test_gender_y = data_gender_y[train_num:]
    train_age_y = data_age_y[:train_num]
    test_age_y = data_age_y[train_num:]

    # create wideResNet model
    model = WideResNet(image_size, depth=depth, k=k)()
    opt = SGD(lr=lr, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt, loss=["categorical_crossentropy", "categorical_crossentropy"], metrics=['accuracy'])

    model.count_params()
    model.summary()

    callbacks = [LearningRateScheduler(schedule=Schedule(epochs_num, lr)),
                 ModelCheckpoint(str(model_path) + "/weights.hdf5", monitor="val_loss", verbose=1, save_best_only=True, mode="auto")]

    # train model
    model.fit(train_x, [train_gender_y, train_age_y], batch_size=batch_size, epochs=epochs_num, callbacks=callbacks,
                         validation_data=(test_x, [test_gender_y, test_age_y]))

    return


if __name__ == '__main__':
    main()
