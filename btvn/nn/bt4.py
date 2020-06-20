from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols, img_chanel = 32, 32, 3

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

print(x_train[0].shape, len(x_train[0].shape))
print(x_train[0].shape[0],x_train[0].shape[1],x_train[0].shape[2])

for i in range(9):
    	# define subplot
	plt.subplot(330 + 1 + i)
	# plot raw pixel data
	plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
# show the figure
plt.show()


# x_train[0].shape, y_test[0]

x_train = x_train.reshape(-1, 1024*3)
x_test = x_test.reshape(-1, 1024*3)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# normalize (0-1)
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
print(x_train[0])
print(y_train[0])
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_train[0])


def trainingModel():
    model = Sequential()

    model.add(Dense(512, activation='relu', input_shape=(1024*3,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()
    model.compile(loss='categorical_crossentropy',
                optimizer=SGD(), # adam, .... gradient descent
                metrics=['accuracy'])
    history = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))

    print(K.eval(model.optimizer.lr))

    return history

plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'], loc='upper left')
history = trainingModel()
plt.plot(history.history['loss'])
plt.show()