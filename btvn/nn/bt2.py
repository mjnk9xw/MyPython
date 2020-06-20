from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train[0].shape, y_test[0]
print(len(x_train[0].shape),x_train[0].shape)
print(x_train[0].shape[0],x_train[0].shape[1])

x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# normalize (0-1)
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
print(y_train[0])
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_train[0])

def changeHyperParameter(rate,activation_name,number_hidden_layer, number_node):
    model = Sequential()
    if(number_hidden_layer == 1):
        if (len(number_node) > 0) :
            model.add(Dense(number_node[0], activation=activation_name, input_shape=(784,)))
            model.add(Dense(number_node[1], activation=activation_name))
            model.add(Dense(num_classes, activation='softmax'))
        else:
            model.add(Dense(512, activation=activation_name, input_shape=(784,)))
            model.add(Dense(32, activation=activation_name))
            model.add(Dense(num_classes, activation='softmax'))
    if(number_hidden_layer == 0):
        model.add(Dense(512, activation=activation_name, input_shape=(784,)))
        model.add(Dense(num_classes, activation='softmax'))
    if(number_hidden_layer == 4):
        model.add(Dense(512, activation=activation_name, input_shape=(784,)))
        model.add(Dense(256, activation=activation_name))
        model.add(Dense(128, activation=activation_name))
        model.add(Dense(64, activation=activation_name))
        model.add(Dense(32, activation=activation_name))
        model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                optimizer=SGD(learning_rate=rate), # adam, .... gradient descent
                metrics=['accuracy'])
    history = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))

    print(K.eval(model.optimizer.lr))

    return history

fig, axs = plt.subplots(4)

learning_rates = [0.0001,0.001,0.01]
axs[0].set_ylabel('loss')
axs[0].set_xlabel('epoch')
axs[0].legend([str(0.0001),str(0.001),str(0.01)], loc='upper left')
for learning_rate in learning_rates:
    history = changeHyperParameter(learning_rate,'relu',1,[])
    axs[0].plot(history.history['loss'])

activation_names = ['relu','tanh','sigmoid']
axs[1].set_ylabel('loss')
axs[1].set_xlabel('epoch')
axs[1].legend(activation_names, loc='upper left')
for activation_name in activation_names:
    history = changeHyperParameter(0.01,activation_name,1,[])
    axs[1].plot(history.history['loss'])

number_hidden_layer = [0,1,4]
axs[2].set_ylabel('loss')
axs[2].set_xlabel('epoch')
axs[2].legend([str(0),str(1),str(4)], loc='upper left')
for number in number_hidden_layer:
    history = changeHyperParameter(0.01,'relu',number,[])
    axs[2].plot(history.history['loss'])

number_node = [[512,32],[512,64],[256,32],[256,64]]
axs[3].set_ylabel('loss')
axs[3].set_xlabel('epoch')
axs[3].legend([str(512)+"-"+str(32),str(512)+"-"+str(64),str(256)+"-"+str(32),str(256)+"-"+str(64)], loc='upper left')
for number in number_node:
    history = changeHyperParameter(0.01,'relu',1,number)
    axs[3].plot(history.history['loss'])

plt.show()