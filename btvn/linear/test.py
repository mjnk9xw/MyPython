import numpy as np
import matplotlib.pyplot as plt
def fx(x):
    np.random.rand(0,3)
    return 3*x*x - 8*x + 3000


x = np.linspace(-2,2,10)
X0 = np.ones((10,1))
x1_matrix = x.reshape(-1,1)
x2_matrix = (x**2).reshape(-1,1)
x_matrix = np.hstack((X0,x1_matrix))
x_matrix = np.hstack((x_matrix,x2_matrix))
# y = fx(x)
# print(x,y)

# plt.scatter(x[:],y[:])
# plt.show()
w = np.array([1.,-1.,1.]).reshape(-1,1)
print(w)
print(w[0:2])
print(x_matrix)
print(x_matrix[:,0:2])