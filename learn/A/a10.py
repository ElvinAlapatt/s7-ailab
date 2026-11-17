"""
A10. Make a 300x300 image from scratch: left half black, right half white, a gray square in the middle. Then a black-to-white gradient (np.tile(np.arange(256), (256,1))).
"""

import cv2
import numpy as np

img = np.zeros((300,300),dtype='uint8')

img[::,150:301:1] = 255

pt1 = (int(0.25*300),int(0.25*300))
pt2 = (int(0.75*300),int(0.75*300))

cv2.rectangle(img,pt1,pt2,127,2)

array = np.linspace(0,255,150) #creates a 150*150 
#this created a one dimensional gradient from 0 to 255
#here the one dimensional array is copied to every single line using slicing
img[int(0.25*300):int(0.75*300):,int(0.25*300):int(0.75*300):] = array


#to create a gradient use linspace and then tile function
row = np.linspace(0,255,300,dtype='uint8')
img2 = np.tile(row , (300,1))
# img2 = np.arange(0,256,1,'uint8')


cv2.imshow("img",img)
cv2.imshow("gradient",img2)
cv2.waitKey(0)
cv2.destroyAllWindows()