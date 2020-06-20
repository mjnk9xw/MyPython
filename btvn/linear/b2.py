import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('btvn/linear/data_linear.csv').values

fig, axs = plt.subplots(2)
fig.suptitle('Linear')

x = data[:, 0]
n = len(data)
X0 = np.ones((n,1))
x_matrix = x.reshape(-1,1)
x_matrix = np.hstack((X0,x_matrix))
y = data[:, 1]
y_matrix = y.reshape(-1,1)
axs[0].scatter(x, y)

numberN = 3000
rate = 0.00001
threshol = 0.001
w = np.array([0.,1.]).reshape(-1,1)
points = [0] * numberN
epoch = [0] * numberN

# f(x) = x*w1 + w0
i = 0
for i in range(numberN):
    predictions = np.dot(x_matrix,w)
    error =  predictions - y_matrix
    points[i] = 1/2 * 1/n * np.sum(error**2) 

    gradient = np.dot(x_matrix.T,error)/n

    w -= gradient*rate
    epoch[i] = i
    print(epoch[i],points[i],w)

    if abs(points[i]) <= threshol and i > 2:
        print('đến ngưỡng dừng: ',i)
        break

# loss func vs epoch
axs[1].plot(epoch[0:i],points[0:i])

# draw line linear regression
xLine = [22,120]
yLine = xLine*w[1] + w[0]
axs[0].plot(xLine,yLine)
plt.show()