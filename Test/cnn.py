from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
#create model
model = Sequential()
#add model layers
model.add(Conv2D(64, kernel_size=5, activation='relu', input_shape=(28,28,1)))
# model.add(Flatten())
# model.add(Dense(10, activation='softmax'))