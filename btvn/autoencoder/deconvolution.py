import numpy as np

# tạo array 
i = np.array(list(range(1, 50)), dtype=np.float).reshape((7, 7))
k = np.array(list(range(1, 10)), dtype=np.float).reshape((3, 3))

print("Input:"); print(i)
print("Kernel:"); print(k)

# convolution với p=1, s=1
def conv2d(i, k):
    o = np.zeros_like(i)
    i = np.pad(i, (1, 1), mode="constant")
    for oy in range(7):
        for ox in range(7):
            for ky in range(3):
                for kx in range(3):
                    o[oy, ox] += k[ky, kx] * i[oy + ky, ox + kx]
    return o

o = conv2d(i, k)
print("convolution:")
print(o)

# deconvolution
def deconv2d(i, k):
    o = np.zeros((i.shape[0] + 2, i.shape[1] + 2))
    for oy in range(7):
        for ox in range(7):
            for ky in range(3):
                for kx in range(3):
                    o[oy + ky, ox + kx] += k[ky, kx] * i[oy, ox]
    return o

print("deconvolution:")
print(deconv2d(o, k))

print("deconvolution (without padding):")
print(deconv2d(o, k)[1:-1,1:-1])