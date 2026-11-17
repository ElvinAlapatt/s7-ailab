import cv2
import numpy as np

img = cv2.imread("dataset/campus2.jpeg")

#log transform there is c the scaling factor and the transformation formula

#it needs image in grayscale
img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

h , w = img.shape

logimg = np.zeros((h,w),dtype='uint8')
#c = 255/log(256)
#c*log(1+r)

c = 255/(np.log(256))

for i in range(h):
    for j in range(w):
        r = img[i,j]
        y = c * np.log(1+r)
        logimg[i,j] = y

cv2.imshow("img",img)
cv2.imshow("logimg",logimg)
cv2.waitKey(0)
cv2.destroyAllWindows()