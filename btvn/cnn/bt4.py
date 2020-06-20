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
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization

batch_size = 64
num_classes = 10
epochs = 25

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

# x_train = x_train.reshape(-1, 1024*3)
# x_test = x_test.reshape(-1, 1024*3)
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
print(x_train.shape[1:])


# Define Model VGG16
def base_model():
    
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same',activation='relu', input_shape=x_train.shape[1:], name='block1_conv1'))
    model.add(Conv2D(64, (3, 3), padding='same',activation='relu', name='block1_conv2'))
   
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    model.add(Conv2D(128, (3, 3), padding='same',activation='relu', name='block2_conv1'))
    model.add(Conv2D(128, (3, 3), padding='same',activation='relu', name='block2_conv2'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    model.add(Conv2D(256, (3, 3), padding='same',activation='relu', name='block3_conv1'))
    model.add(Conv2D(256, (3, 3), padding='same',activation='relu', name='block3_conv2'))
    model.add(Conv2D(256, (3, 3), padding='same',activation='relu', name='block3_conv3'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    model.add(Conv2D(512, (3, 3), padding='same',activation='relu', name='block4_conv1'))
    model.add(Conv2D(512, (3, 3), padding='same',activation='relu', name='block4_conv2'))
    model.add(Conv2D(512, (3, 3), padding='same',activation='relu', name='block4_conv3'))
   
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    model.add(Conv2D(512, (3, 3), padding='same',activation='relu', name='block5_conv1'))
    model.add(Conv2D(512, (3, 3), padding='same',activation='relu', name='block5_conv2'))
    model.add(Conv2D(512, (3, 3), padding='same',activation='relu', name='block5_conv3'))
   
    model.add(Flatten())

    model.add(Dense(4096,activation='relu',name='fc1'))
    model.add(Dense(4096, name='fc2',activation='relu'))

    model.add(Dense(num_classes,activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.0005), metrics=['accuracy'])
    return model

cnn_n = base_model()
cnn_n.summary()
history = cnn = cnn_n.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))

plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'], loc='upper left')
plt.plot(history.history['loss'])
plt.show()