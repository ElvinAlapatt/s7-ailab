import numpy as np
import cv2

img = cv2.imread('dataset/campus2.jpeg')

# gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

# _ , binary = cv2.threshold(gray , 127 , 255 , cv2.THRESH_BINARY)

#eqn = 0.114*b + 0.587*g + 0.299*r 

h , w , _ = img.shape

gray = np.zeros((h,w),dtype='uint8')
binary = np.zeros((h,w),dtype='uint8')

for i in range(h):
    for j in range(w):
        b , g , r = img[i,j]
        gray[i,j] = 0.114*b + 0.587*g + 0.299*r
        if gray[i,j] <= 127:
            binary[i,j] = 0
        else:
            binary[i,j] = 255


cv2.imshow("og",img)
cv2.imshow("gray",gray)
cv2.imshow("bin",binary)
cv2.waitKey(0)
cv2.destroyAllWindows()