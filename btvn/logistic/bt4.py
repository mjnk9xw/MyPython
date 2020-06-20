import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Hàm sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Hàm sigmoid
def xaxb(x,w):
    return (x[0] - w[0]) / (x[0] - w[1]) - x[1]

data = pd.read_csv('btvn/logistic/dataset_xor.csv').values
N, d = data.shape
x = data[:, 0:d-1].reshape(-1, d-1)
y = data[:, 2].reshape(-1, 1)
print(x,y)
print(y[:,0]==1)

# Vẽ data bằng scatter
x_cho_vay = x[y[:,0]==1]
x_tu_choi = x[y[:,0]==0]
print(x_cho_vay)
print(x_tu_choi)


# # Thêm cột 1 vào dữ liệu x
# x = np.hstack((np.ones((N, 1)), x))
# w = np.array([0.,0.1,0.1]).reshape(-1,1)

# # Số lần lặp bước 2
# numOfIteration = 1000
# cost = np.zeros((numOfIteration,1))
# learning_rate = 0.001

# for i in range(numOfIteration):
#     # Tính giá trị dự đoán
#     y_predict = sigmoid(np.dot(x, w))
#     print('y = ' ,y_predict)
#     cost[i] = -np.sum(np.multiply(y, np.log(y_predict)) + np.multiply(1-y, np.log(1-y_predict)))
#     # Gradient descent
#     w = w - learning_rate * np.dot(x.T, y_predict-y)	 
#     print('cost = ',cost[i])

# # Vẽ đường phân cách.
# t = 0.5
# plt.scatter(x_cho_vay[:, 0], x_cho_vay[:, 1], c='red', edgecolors='none', s=30, label='cho vay')
# plt.scatter(x_tu_choi[:, 0], x_tu_choi[:, 1], c='blue', edgecolors='none', s=30, label='từ chối')
# plt.legend(loc=1)
# plt.xlabel('mức lương (triệu)')
# plt.ylabel('kinh nghiệm (năm)')
# plt.plot((10, 20),(-(w[0]+10*w[1]+ np.log(1/t-1))/w[2], -(w[0] + 20*w[1]+ np.log(1/t-1))/w[2]), 'g')
# plt.show()

# Thêm cột 1 vào dữ liệu x
# x = np.hstack((np.ones((N, 1)), x))
w = np.array([0.1,0.1]).reshape(-1,1)

# Số lần lặp bước 2
numOfIteration = 1000
cost = np.zeros((numOfIteration,1))
learning_rate = 0.001

for i in range(numOfIteration):
    # Tính giá trị dự đoán
    y_predict = sigmoid(xaxb(x,w))
    print('y = ' ,y_predict)
    cost[i] = -np.sum(np.multiply(y, np.log(y_predict)) + np.multiply(1-y, np.log(1-y_predict)))
    # Gradient descent
    w = w - learning_rate * np.dot(x.T, y_predict-y)
    # w[0] = w[0] - learning_rate*np.sum(w[0]/(x[:,0]+w[1]))	 
    # w[1] = w[1] - learning_rate*np.sum((w[0]*x[:,0])/((x[:,0]+w[1])**2))	 
    print('cost = ',cost[i])
    print('w = ',w)

# Vẽ đường phân cách.
t = 0.5
plt.scatter(x_cho_vay[:, 0], x_cho_vay[:, 1], c='red', edgecolors='none', s=30, label='cho vay')
plt.scatter(x_tu_choi[:, 0], x_tu_choi[:, 1], c='blue', edgecolors='none', s=30, label='từ chối')
plt.legend()
plt.xlabel('mức lương (triệu)')
plt.ylabel('kinh nghiệm (năm)')
print(w[0],w[1])

def fxYLine(x1,w0,w1):
    return (x1 - w0) / (x1 - w1)

xLine1 = np.linspace(start = -0.5,stop = 2,num =50)
# xLine2 = np.linspace(start = -0.5,stop = 0,num =50)

yLine1 = [0]*50
# yLine2 = [0]*50
for i in range(50):
    yLine1[i] = fxYLine(xLine1[i],w[0],w[1])
    # yLine2[i] = fxYLine(xLine2[i],w[0],w[1])
    if i < 5:
        print(xLine1[i],yLine1[i])

plt.plot(xLine1[:],yLine1[:], 'g')
# plt.plot(xLine2[:],yLine2[:], 'g')
plt.show()