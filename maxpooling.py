import cv2
import numpy as np

img = cv2.imread('dataset/campus.jpeg',0)

imgmax = np.zeros((img.shape[0],img.shape[1]),dtype='uint8')
imgmed = np.zeros((img.shape[0],img.shape[1]),dtype='uint8')

h , w  = img.shape

n = 3

pad = n//2

for i in range(pad , h - pad ):
    for j in range(pad , w - pad):
        kernal = img[i-pad:i+pad+1,j-pad:j+pad+1]
        med = np.median(kernal)
        max = np.max(kernal)

        imgmed[i,j] = med 
        imgmax[i,j] = max

cv2.imshow("og",img)
cv2.imshow("max",imgmax)
cv2.imshow("median",imgmed)
cv2.waitKey(0)
cv2.destroyAllWindows()