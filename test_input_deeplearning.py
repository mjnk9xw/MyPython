''' 
1. Dùng thư viện pandas load file data_linear.csv, vẽ đồ thị quan hệ diện tích và giá nhà.
'''
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data_linear.csv').values

plt.xlabel('Diện tích')
plt.ylabel('Giá')
x = data[:, 0]
y = data[:, 1]
plt.plot(x, y,'go-')
plt.show()

'''
2. Dùng thư viện opencv load ảnh image.png và thực hiện các bước sau:
- Cắt góc phần tư trái trên của ảnh.
- Resize ảnh, dài rộng còn một nửa.
- Thực hiện Gaussian blur ảnh.
- Phát hiện edge trong ảnh.
'''
import cv2
import numpy as np

IMG_PATH = 'image.png'
img = cv2.imread(IMG_PATH)
print(img.shape)
h = img.shape[0]
w = img.shape[1]
crop_img = img[0:h//2,0:w//2,:]
cv2.imshow("1/4", crop_img)

img_resized = cv2.resize(src=img, dsize=(h//2, w//2))
cv2.imshow("1/2", img_resized)

blur_img = cv2.GaussianBlur(img, ksize=(9, 9), sigmaX=4, sigmaY=4)
cv2.imshow("GaussianBlur", blur_img)

edges = cv2.Canny(img, 100, 200)
cv2.imshow('edge', edges)

cv2.waitKey(0)
