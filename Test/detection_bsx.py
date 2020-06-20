from cv2 import cv2
import numpy as np
from skimage.morphology import opening
 
image = cv2.imread("data/1.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_,thresh = cv2.threshold(gray,127,225,cv2.THRESH_TOZERO)
cv2.imshow("thresh",thresh)

element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(5, 11))

morph_img = thresh.copy()
cv2.morphologyEx(src=thresh, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img)
cv2.imshow("morph_img",morph_img)


contours,_ = cv2.findContours(morph_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    print(c)
    r = cv2.boundingRect(c)
    cv2.rectangle(image,(r[0],r[1]),(r[0]+r[2],r[1]+r[3]),(0,0,255),2)

cv2.imshow("img",image)

cv2.waitKey(0)
cv2.destroyAllWindows()