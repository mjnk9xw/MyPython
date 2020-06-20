import numpy as np
import matplotlib.pyplot as plt

# f(x) = x^2
def grad(x):
    return 2*x

n = 100
def xValues(x, learning_rate):
    r = np.zeros((n,2))
    for i in range(n):
        r[i][0] = i
        r[i][1] = x**2
        x -= learning_rate*grad(x)
        print(x)
        # if abs(grad(x)) < 1e-3:
        #     return r, i
    return r , n

x = -5
rate = [0.0007,0.007,0.07,0.7]
for learning_rate in rate:
    points, epoch_n = xValues(x,learning_rate)
    plt.plot(points[0:epoch_n,0],points[0:epoch_n,1],label=str(learning_rate))
    print(learning_rate)

plt.xlabel('epoch #')
plt.ylabel('loss')
plt.legend()
plt.show()