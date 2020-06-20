import numpy as np
from cv2 import cv2
from math import sqrt
import matplotlib.pyplot as plt

img = cv2.imread('btvn/cnn/image.png',1)

img2 = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )

img3 = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )

height = np.size(img, 0)

width = np.size(img, 1)

kernel_mat=(np.array([[1,2,1],[0,0,0],[-1,-2,-1]]))

for i in range (1,height-1):

    for j in range (1,width-1):

        # x=img2[i+1,j-1]*kernel_mat[2,0]+img2[i+1,j]*kernel_mat[2,1]+img2[i+1,j+1]*kernel_mat[2,2]+(img2[i-1,j-1]*kernel_mat[0,0]+img2[i-1,j]*kernel_mat[0,1]+img2[i-1,j+1]*kernel_mat[0,2])

        y=(img2[i-1,j+1]*kernel_mat[0,2]+img2[i,j+1]*kernel_mat[1,2]+img2[i+1,j+1]*kernel_mat[2,2])+(img2[i-1,j-1]*kernel_mat[0,0]+img2[i,j-1]*kernel_mat[1,0]+img2[i+1,j-1]*kernel_mat[2,0])

        x = 0
        # print(x, y)

        img3[i-1,j-1]=sqrt(x**2+y**2)
        
# contour,hier=cv2.findContours(img3,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img3,contour,-1,(255,0,0),1)

contour,hier=cv2.findContours(img3,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img3,contour,-1,(255,0,0),1)

plt.imshow(img3,cmap='gray')
plt.show()