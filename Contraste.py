import numpy as np
import cv2
import math

img = cv2.imread('cr7.jpg',0)
cv2.imshow('imagen original',img)

height = img.shape[0]
width = img.shape[1]

contrast = 1.3

for i in np.arange(height):
    for j in np.arange(width):
        a = img.item(i,j)
        b = math.ceil(a * contrast)
        if b > 255:
            b = 255
        img.itemset((i,j), b)

cv2.imshow('Realce Contraste', img)

cv2.waitKey(0)
cv2.destroyAllWindows()