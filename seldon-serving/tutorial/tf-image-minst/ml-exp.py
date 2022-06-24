import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

import mlflow
import mlflow.tensorflow

mlflow.start_run()

mlflow.tensorflow.autolog(log_input_examples=True, log_model_signatures=True)

(X_trian, y_train), (X_test, y_test) = load_data()

X_train = X_trian.reshape(X_trian.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_trian.shape, y_train.shape, X_test.shape, y_test.shape

train_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.9, 1.1],
    rotation_range=10,
    rescale=1.0 / 255,
)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_dataset = train_datagen.flow(X_train, y_train, batch_size=128)
val_dataset = val_datagen.flow(X_test, y_test, batch_size=128)

n_step_train = X_train.shape[0] // 128
n_step_val = X_test.shape[0] // 128

fig, ax = plt.subplots(3, 3, figsize=(15, 10))

X, y = train_dataset.next()

for i in range(3):
    for j in range(3):
        ax[i, j].imshow(X[i * 3 + j].reshape(28, 28), cmap="gray")
        ax[i, j].set_title(y[i * 3 + j])

        ax[i, j].axis("off")

mlflow.log_figure(fig, "train-batch1.png")

fig, ax = plt.subplots(3, 3, figsize=(15, 10))

X, y = val_dataset.next()

for i in range(3):
    for j in range(3):
        ax[i, j].imshow(X[i * 3 + j].reshape(28, 28), cmap="gray")
        ax[i, j].set_title(y[i * 3 + j])

        ax[i, j].axis("off")

mlflow.log_figure(fig, "val-batch1.png")


model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(1, 3, (1, 1), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, 3, (1, 1), padding="same"),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(2),
        tf.keras.layers.Conv2D(64, 3, (1, 1), padding="same"),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["acc"],
)

model.fit(
    train_dataset,
    epochs=10,
    steps_per_epoch=n_step_train,
    validation_data=val_dataset,
    validation_steps=n_step_val,
)

mlflow.end_run()
