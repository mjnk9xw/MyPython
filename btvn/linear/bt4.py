import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('btvn/linear/data_linear.csv').values

fig, axs = plt.subplots(3)
fig.suptitle('Loss')

x = data[:, 0]
n = len(data)
X0 = np.ones((n,1))
x_matrix = x.reshape(-1,1)
x_matrix = np.hstack((X0,x_matrix))
y = data[:, 1]
y_matrix = y.reshape(-1,1)

numberN = 100
rates = [0.000001,0.00001,0.0001]
rate_index = 0
for rate in rates:
    # f(x) = x*w1 + w0
    threshol = 0.01
    w = np.array([0.,1.]).reshape(-1,1)
    points = [0] * numberN
    epoch = [0] * numberN
    i = 0
    for i in range(numberN):
        yy = np.dot(x_matrix,w) - y_matrix

        points[i] = 1/2 * 1/n * np.sum(yy**2)
        
        w[0] -= rate*np.sum(yy)
        w[1] -= rate*np.sum(np.multiply(x_matrix[:,1].reshape(-1,1),yy))
        
        epoch[i] = i
        # print(epoch[i],points[i],w[1],w[0])

    # loss func vs epoch
    axs[rate_index].plot(epoch[0:i],points[0:i],label=str(rate))
    axs[rate_index].legend()
    rate_index += 1
plt.show()