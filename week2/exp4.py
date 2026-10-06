import cv2
import numpy as np

img_path1 = "/mnt/d/ailab/week2/car1.jpg"
img_path2 = "/mnt/d/ailab/week2/car2.jpg" 

img1 = cv2.imread(img_path1)
img2 = cv2.imread(img_path2)

#resized the img2 to the width(img1.shape[1])  and height(img1.shape[0])
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

added_img = cv2.add(img1,img2)

subtracted_img = cv2.subtract(img1,img2)

alpha = 0.7
beta = 0.3
gamma = 0 #offset
blended_img = cv2.addWeighted(img1,alpha,img2,beta,gamma)

cv2.imwrite("exp4added.jpg",added_img)
cv2.imwrite("exp4sub.jpg",subtracted_img)
cv2.imwrite("exp4blended.jpg",blended_img)
