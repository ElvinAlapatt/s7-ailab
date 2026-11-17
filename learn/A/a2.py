'''
A2. Split the image into B, G, R channels. Display each channel in grayscale. Then build an image where only the red channel is kept (others zero).
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt 

img_path = "dataset/campus2.jpeg"

img = cv2.imread(img_path)

h , w , _ = img.shape

bimg = np.zeros((h,w),dtype='uint8')
gimg = np.zeros((h,w),dtype='uint8')
rimg  = np.zeros((h,w),dtype='uint8')

imgnew = np.zeros((h,w,3),dtype='uint8')

for i in range(h):
    for j in range(w):
        t = img[i,j] # it is the tuple of 3 colors BGR
        bimg[i,j] = t[0]
        gimg[i,j] = t[1]
        rimg[i,j] = t[2]
        imgnew[i,j] = (0,0,t[2])
cv2.imshow("bimg",bimg)
cv2.imshow("gimg",gimg)
cv2.imshow("rimg",rimg)

cv2.imshow("imgnew",imgnew)

cv2.waitKey(0)
cv2.destroyAllWindows()