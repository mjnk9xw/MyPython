from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten,Input,Add,GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

batch_size = 64
num_classes = 10
epochs = 25

# input image dimensions
img_rows, img_cols, img_chanel = 32, 32, 3

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# print(x_train[0].shape, len(x_train[0].shape))
# print(x_train[0].shape[0],x_train[0].shape[1],x_train[0].shape[2])

# for i in range(9):
#     	# define subplot
# 	plt.subplot(330 + 1 + i)
# 	# plot raw pixel data
# 	plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
# # show the figure
# plt.show()


# x_train[0].shape, y_test[0]

# x_train = x_train.reshape(-1, 1024*3)
# x_test = x_test.reshape(-1, 1024*3)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# normalize (0-1)
x_train /= 255
x_test /= 255
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# print(x_train[0])
# print(y_train[0])
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# print(y_train[0])
# print(x_train.shape[1:])

inputs = Input(x_train.shape[1:])
x = Conv2D(16, (3, 3), padding='same')(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)

n = 9 # 56 layers
channels = [16, 32, 64]
# Define Model resnet
for c in channels:
    for i in range(n):
        print('layer = ', c, i)
        subsampling = i == 0 and c > 16
        strides = (2, 2) if subsampling else (1, 1)
        y = Conv2D(c, kernel_size=(3, 3), padding="same", strides=strides)(x)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Conv2D(c, kernel_size=(3, 3), padding="same")(y)
        y = BatchNormalization()(y)        
        if subsampling:
            x = Conv2D(c, kernel_size=(1, 1), strides=(2, 2), padding="same")(x)
            print('subsampling = ',c,i)
        x = Add()([x, y])
        x = Activation('relu')(x)

# x = GlobalAveragePooling2D()(x)
x = Flatten()(x)
outputs = Dense(num_classes,activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)

model.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])

model.summary()
history = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=2,
            validation_data=(x_test, y_test))

plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss','val_loss','accuracy', 'val_accuracy'], loc='upper left')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()