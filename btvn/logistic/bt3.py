import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Hàm sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

data = pd.read_csv('btvn/logistic/dataset.csv').values
N, d = data.shape
x = data[:, 0:d-1].reshape(-1, d-1)
y = data[:, 2].reshape(-1, 1)
# print(x,y)
# print(y[:,0]==1)

# Vẽ data bằng scatter
x_cho_vay = x[y[:,0]==1]
x_tu_choi = x[y[:,0]==0]
# print(x_cho_vay)
# print(x_tu_choi)

# plt.scatter(x_cho_vay[:, 0], x_cho_vay[:, 1], c='red', edgecolors='none', s=30, label='cho vay')
# plt.scatter(x_tu_choi[:, 0], x_tu_choi[:, 1], c='blue', edgecolors='none', s=30, label='từ chối')
# plt.legend(loc=1)
# plt.xlabel('mức lương (triệu)')
# plt.ylabel('kinh nghiệm (năm)')

# Thêm cột 1 vào dữ liệu x
x = np.hstack((np.ones((N, 1)), x))
w = np.array([0.,0.1,0.1]).reshape(-1,1)

# Số lần lặp bước 2
numOfIteration = 1000
cost = np.zeros((numOfIteration,1))
learning_rates = [0.0001,0.001,0.01,0.025]

for learning_rate in learning_rates:

    points = np.zeros((numOfIteration,2))
    for i in range(numOfIteration):
        # Tính giá trị dự đoán
        y_predict = sigmoid(np.dot(x, w))
        print(y_predict)
        print(w)
        cost[i] = -np.sum(np.multiply(y, np.log(y_predict)) + np.multiply(1-y, np.log(1-y_predict)))
        # Gradient descent
        w = w - learning_rate * np.dot(x.T, y_predict-y)	 
        # print(cost[i])

        points[i][0] = i
        points[i][1] = cost[i]

    plt.plot(points[:,0],points[:,1],label=str(learning_rate))
    plt.legend()
    print(learning_rate)

# Vẽ đường phân cách.
# t = 0.5
# plt.scatter(x_cho_vay[:, 0], x_cho_vay[:, 1], c='red', edgecolors='none', s=30, label='cho vay')
# plt.scatter(x_tu_choi[:, 0], x_tu_choi[:, 1], c='blue', edgecolors='none', s=30, label='từ chối')
# plt.legend(loc=1)
# plt.xlabel('mức lương (triệu)')
# plt.ylabel('kinh nghiệm (năm)')
# plt.plot((4, 10),(-(w[0]+4*w[1]+ np.log(1/t-1))/w[2], -(w[0] + 10*w[1]+ np.log(1/t-1))/w[2]), 'g')
plt.xlabel('epoch #')
plt.ylabel('loss')
plt.show()