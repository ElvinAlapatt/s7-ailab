import cv2
import numpy as np

img1_path = '/mnt/d/ailab/week2/img1.jpeg' #circle
img2_path = '/mnt/d/ailab/week2/img2.jpeg' # rectangle

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

img2 = cv2.resize(img2, (img1.shape[1],img1.shape[0]))

and_img = cv2.bitwise_and(img1,img2)
or_img = cv2.bitwise_or(img1,img2)
xor_img = cv2.bitwise_xor(img1,img2)

cv2.imwrite("and_img.jpg",and_img)
cv2.imwrite("or_img.jpg",or_img)
cv2.imwrite("xor_img.jpg",xor_img)
