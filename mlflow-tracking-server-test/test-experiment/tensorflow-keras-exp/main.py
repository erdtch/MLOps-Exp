import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras as tfk
from tensorflow.keras.datasets.mnist import load_data


import mlflow

mlflow.set_tracking_uri("http://192.168.1.12:5000")
mlflow.tensorflow.autolog()

(X_train, y_train), (X_test, y_test) = load_data()

X_train = (X_train / 255.0).reshape(-1, 28, 28, 1)
X_test = (X_test / 255.0).reshape(-1, 28, 28, 1)

model = tfk.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(1, 3, (1, 1), padding='same'),
    layers.BatchNormalization(),

    layers.Conv2D(32, 3, (1, 1), padding='same'),
    layers.LeakyReLU(),
    layers.BatchNormalization(),
    layers.MaxPool2D(2),

    layers.Conv2D(64, 3, (1, 1), padding='same'),
    layers.LeakyReLU(),
    layers.BatchNormalization(),

    layers.Flatten(),
    
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=tfk.optimizers.Adam(learning_rate=0.0001),
    loss=tfk.losses.SparseCategoricalCrossentropy(),
    metrics=['acc']
)


print('-'*20)
print("Starting Training")
print("Input shape", X_train.shape, X_test.shape)
print('-'*20)

model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))