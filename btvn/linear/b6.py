import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fx(w2,w1,w0,x):
    return (x**2)*w2 + x*w1 + w0

data = pd.read_csv('btvn/linear/data_square.csv').values

# test data
# def fxtest(x):
#     np.random.rand(0,3)
#     return 3*x*x + 8*x + 3000

# x = np.linspace(-2,50,100)
# y = fxtest(x)
# n = 100

plt.xlabel('Diện tích')
plt.ylabel('Giá')
x = data[:, 0]
n = len(data)
X0 = np.ones((n,1))
x1_matrix = x.reshape(-1,1)
x2_matrix = (x**2).reshape(-1,1)
x_matrix = np.hstack((X0,x1_matrix))
x_matrix = np.hstack((x_matrix,x2_matrix))
y = data[:, 1]
y_matrix = y.reshape(-1,1)
plt.scatter(x, y)

# numberN = 1000
# rate = 0.1e-20
# threshol = 0.01
# w = np.array([2000.,-100.,1.]).reshape(-1,1)

numberN = 10000
rate = 0.00001
w = np.array([1.,-1.,1.]).reshape(-1,1)
points = [0] * numberN
# f(x) = x^2*w2 + x*w1 + w0
for i in range(numberN):
    yy = np.dot(x_matrix,w) - y_matrix

    points[i] = 1/2 * 1/n * np.sum(yy**2)
    
    w[0] -= rate*np.sum(yy)
    w[1] -= rate*np.sum(np.multiply(x_matrix[:,1].reshape(-1,1),yy))
    # w[2] -= rate*np.sum(np.multiply(x_matrix[:,2].reshape(-1,1),yy)) 
    print((y_matrix - np.dot(x_matrix[:,0:2],w[0:2])))
    w[2] -= rate*np.sum( (y_matrix - np.dot(x_matrix[:,0:2],w[0:2])) / x2_matrix)

    print('point: ',points[i])
    print('sum :', np.sum(yy))
    print('w: ',w[0],w[1],w[2])
    print('-----------------------------------------------')

# x = 22m
# x = 120m
xLine = np.linspace(start = 26,stop = 120,num =100)
plt.plot(xLine,fx(w[2],w[1],w[0],xLine))
plt.show()

