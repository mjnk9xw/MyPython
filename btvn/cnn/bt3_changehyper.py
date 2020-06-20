from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
from keras.utils import np_utils
from keras.datasets import mnist
import kerastuner
import numpy as np
import matplotlib.pyplot as plt

# 2. Load dữ liệu MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_val, y_val = X_train[50000:60000,:], y_train[50000:60000]
X_train, y_train = X_train[:50000,:], y_train[:50000]
print(X_train.shape)

print(X_val.shape)

# 3. Reshape lại dữ liệu cho đúng kích thước mà keras yêu cầu
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train/255.
X_test = X_test/255.
X_val = X_val/255.
print(X_train.shape)

# 4. One hot encoding label (Y)
Y_train = np_utils.to_categorical(y_train, 10)
Y_val = np_utils.to_categorical(y_val, 10)
Y_test = np_utils.to_categorical(y_test, 10)
print('Dữ liệu y ban đầu ', y_train[0])
print('Dữ liệu y sau one-hot encoding ',Y_train[0])


def build_model(hp):
    model = keras.Sequential()

    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28,28,1)))

    # Thêm Convolutional layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))

    # Thêm Max pooling layer
    if hp.Choice('pooling_1', ['avg', 'max']) == 'max':
            model.add(layers.MaxPooling2D(pool_size=(2,2)))
    else:
        model.add(layers.AveragePooling2D(pool_size=(2,2)))
    # Flatten layer chuyển từ tensor sang vector
    model.add(layers.Flatten())

    # Thêm Fully Connected layer với 128 nodes và dùng hàm sigmoid
    # relu <-> sigmoid
    # 1. sigmoid is better
    model.add(layers.Dense(128, activation='relu'))

    model.add(layers.Dense(10, activation='softmax'))

    model.compile(
        # optimizer=keras.optimizers.Adam(
        #     hp.Float(
        #         'learning_rate',
        #         min_value=1e-4,
        #         max_value=1e-2,
        #         sampling='LOG',
        #         default=1e-3
        #     )
        # ),
        optimizer= keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


class MyTuner(kerastuner.tuners.BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        # You can add additional HyperParameters for preprocessing and custom training loops
        # via overriding `run_trial`
        kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 64, 64, step=64)
        # kwargs['epochs'] = trial.hyperparameters.Int('epochs', 10, 30)
        super(MyTuner, self).run_trial(trial, *args, **kwargs)

# Uses same arguments as the BayesianOptimization Tuner.
tuner = MyTuner(
    build_model,
    objective='val_accuracy',
    max_trials=1,
    executions_per_trial=1,
    directory='btvn/cnn',
    project_name='tunnermnist',
    overwrite=True)

print('search begin')


tuner.search_space_summary()
tuner.search(X_train, Y_train,
            epochs=10,
            validation_data=(X_val, Y_val))

print('get best models')

best_model = tuner.get_best_models(1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(1)[0]