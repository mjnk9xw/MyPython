from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

classes = np.unique(y_train)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

x_train = x_train.reshape(x_train.shape[0],-1)

pca = PCA()
pca_data = pca.fit_transform(x_train)

print(pca_data)

# plt.scatter(data_pca[:,0],data_pca[:,1])
# plt.show()