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
# n, bins, patches = plt.hist(x=y_val[:], bins='auto', color='#0504aa',
#                             alpha=0.7, rwidth=0.85)
# n, bins, patches = plt.hist(x=y_train[:], bins='auto', color='#607c8e',
#                             alpha=0.7, rwidth=0.85)
n, bins, patches = plt.hist(x=y_test[:], bins='auto', color='#607c8e',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('y')
plt.ylabel('number')
plt.title('Histogram dataset')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.show()