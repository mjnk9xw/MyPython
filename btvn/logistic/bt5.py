import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Hàm sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mapFeature(X1, X2, degree):
    res = np.ones(X1.shape[0])
    for i in range(1,degree + 1):
        for j in range(0,i + 1):
            print('ij = ',i,j)
            res = np.column_stack((res, (X1 ** (i-j)) * (X2 ** j)))
    return res

data = pd.read_csv('btvn/logistic/dataset_xor.csv').values
N, d = data.shape
x = data[:, 0:d-1].reshape(-1, d-1)
y = data[:, 2].reshape(-1, 1)


# Vẽ data bằng scatter
x_cho_vay = x[y[:,0]==1]
x_tu_choi = x[y[:,0]==0]

x = mapFeature(x[:, 0], x[:, 1], 2)
print(x)

# Thêm cột 1 vào dữ liệu x
# x = np.hstack((np.ones((N, 1)), x))
w = np.zeros(x.shape[1]).reshape(x.shape[1], 1)
print(w)

# Số lần lặp bước 2
numOfIteration = 1000
cost = np.zeros((numOfIteration,1))
learning_rate = 0.005
for i in range(numOfIteration):
    # Tính giá trị dự đoán
    y_predict = sigmoid(np.dot(x, w))
    cost[i] = -np.sum(np.multiply(y, np.log(y_predict)) + np.multiply(1-y, np.log(1-y_predict)))
    w = w - learning_rate * np.dot(x.T, y_predict-y)

# Vẽ đường phân cách.
t = 0.5
plt.scatter(x_cho_vay[:, 0], x_cho_vay[:, 1], c='red', edgecolors='none', s=30, label='cho vay')
plt.scatter(x_tu_choi[:, 0], x_tu_choi[:, 1], c='blue', edgecolors='none', s=30, label='từ chối')
plt.legend()
plt.xlabel('mức lương (triệu)')
plt.ylabel('kinh nghiệm (năm)')

def plotDecisionBoundary(theta,degree, axes):
    u = np.linspace(-2, 2, 50)
    v = np.linspace(-2, 2, 50)
    U,V = np.meshgrid(u,v)
    print('u',u)
    print('v',v)
    print('U',U)
    print('V',V)

    U = np.ravel(U)
    V = np.ravel(V)
    Z = np.zeros((len(u) * len(v)))
    
    X_poly = mapFeature(U, V, degree)
    Z = X_poly.dot(theta)

    print('Z',Z)
    
    # reshape U, V, Z back to matrix
    U = U.reshape((len(u), len(v)))
    V = V.reshape((len(u), len(v)))
    Z = Z.reshape((len(u), len(v)))
    
    cs = axes.contour(U,V,Z,levels=[0],cmap= "Greys_r")
    return cs

fig, axes = plt.subplots();
axes.scatter(x_cho_vay[:, 0], x_cho_vay[:, 1], c='red', edgecolors='none', s=30, label='cho vay')
axes.scatter(x_tu_choi[:, 0], x_tu_choi[:, 1], c='blue', edgecolors='none', s=30, label='từ chối')
axes.legend()
axes.set_xlabel('mức lương (triệu)')
axes.set_ylabel('kinh nghiệm (năm)')
plotDecisionBoundary(w,2,axes)

plt.show()