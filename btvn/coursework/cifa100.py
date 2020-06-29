from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, UpSampling2D, Flatten, BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras import optimizers
from keras.datasets import cifar100
import tensorflow as tf
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.transform import resize
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator


num_classes = 100
nb_epochs = 15

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# random data
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                    test_size=0.2,
                                                    random_state=0,
                                                    stratify=y_train)
print('random data = ',x_train.shape,x_val.shape,y_train.shape,y_val.shape)
#Pre-process the data
x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)

# datagen = ImageDataGenerator(preprocessing_function=get_random_eraser(v_l=0, v_h=1, pixel_level=True))
# datagen.fit(x_train)
aug_train = ImageDataGenerator(rescale=1./255, rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, 
                         zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
# augementation cho val
aug_val= ImageDataGenerator(rescale=1./255)

'''
Vẽ biểu đồ dataset
'''
import matplotlib.pyplot as plt
n, bins, patches = plt.hist(x=y_val[:], bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
n, bins, patches = plt.hist(x=y_train[:], bins='auto', color='#607c8e',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('y')
plt.ylabel('number')
plt.title('Histogram dataset')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.show()

y_train = np_utils.to_categorical(y_train, num_classes)
y_val = np_utils.to_categorical(y_val, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in resnet_model.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = True
    else:
        layer.trainable = False

model = Sequential()
model.add(UpSampling2D())
model.add(UpSampling2D())
model.add(UpSampling2D())
model.add(resnet_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(Dropout(.25))
model.add(BatchNormalization())
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

t=time.time()
historytemp = model.fit_generator(aug_train.flow(x_train, y_train,
                                  batch_size=64),
                                  steps_per_epoch=x_train.shape[0] // 64,
                                  epochs=15,
                                  validation_data=(aug_val.flow(x_val, y_val, batch_size=64)),
                                  validation_steps=len(x_val)//64,)
print('Training time: %s' % (t - time.time()))

model.summary()

# !pip install h5py
# model.save('model.h5')
# from google.colab import files
# files.download("model.h5")

new_model = tf.keras.models.load_model('model.h5')

# Check its architecture
# new_model.summary()
x_test = preprocess_input(x_test)
loss, acc = new_model.evaluate(x_test,  y_test, verbose=1)
print('[accuracy_test]: {:5.2f}%'.format(100*acc))

import matplotlib.pyplot as plt
# list all data in history
print(historytemp.history.keys())
# summarize history for accuracy
plt.plot(historytemp.history['accuracy'])
plt.plot(historytemp.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(historytemp.history['loss'])
plt.plot(historytemp.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()