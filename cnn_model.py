import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle


# 0 = cat
# 1 = dog

dir_path = os.getcwd()

def get_datasets():
    train_cats = np.array([cv2.resize(cv2.imread(f"{dir_path}\\dataset\\training_set\\cats\\cat.{i}.jpg"), (128, 128)) for i in range(1, 4001)])  # (4000, 128, 128, 3)
    train_dogs = np.array([cv2.resize(cv2.imread(f"{dir_path}\\dataset\\training_set\\dogs\\dog.{i}.jpg"), (128, 128)) for i in range(1, 4001)])  # (4000, 128, 128, 3)
    test_cats = np.array([cv2.resize(cv2.imread(f"{dir_path}\\dataset\\test_set\\cats\\cat.{i}.jpg"), (128, 128)) for i in range(4001, 5001)])  # (1000, 128, 128, 3)
    test_dogs = np.array([cv2.resize(cv2.imread(f"{dir_path}\\dataset\\test_set\\dogs\\dog.{i}.jpg"), (128, 128)) for i in range(4001, 5001)])  # (1000, 128, 128, 3)
    
    train_images = np.concatenate((train_cats, train_dogs))  # (8000, 128, 128, 3)
    train_labels = np.concatenate((np.zeros((4000,)), np.ones((4000,))))  # (8000,)

    test_images = np.concatenate((test_cats, test_dogs))  # (2000, 128, 128, 3)
    test_labels = np.concatenate((np.zeros((1000,)), np.ones((1000,))))  # (2000,)

    train_images, train_labels = shuffle(train_images, train_labels)
    test_images, test_labels = shuffle(test_images, test_labels)

    return train_images, train_labels, test_images, test_labels

def create_model():
    model = keras.Sequential()
    input_shape = (128, 128, 3)

    model.add(keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=input_shape))
    model.add(keras.layers.AveragePooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(keras.layers. MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024, activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model

def fit_model(model):
    model.fit(train_images, train_labels, epochs=10, batch_size=512)

    result = model.evaluate(test_images, test_labels)

    print(f"result: {result}")

def save_model(model):
    model.save("model.h5")

def load_model():
    return keras.models.load_model("model.h5")


if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = get_datasets()

    # model = create_model()

    # fit_model(model)

    # save_model(model)

    model = load_model()

    model.summary()

    print(model.evaluate(test_images, test_labels))
