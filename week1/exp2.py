import numpy as np
import cv2

#img_path = "D:\ailab\week1\exp1_sample_img.jpg"
img_path = "/mnt/d/ailab/week1/exp1_sample_img.jpg"

img = cv2.imread(img_path)

h , w , _ = img.shape

gray = np.zeros((h,w),dtype='uint8')
binary = np.zeros((h,w),dtype='uint8')


for i in range(h):
	for j in range(w):
		b , g , r = img[i,j]
		y = 0.114*b + 0.587*g + 0.299*r
		
		gray[i,j] = min(max(y,0),255)
		
		if y > 127:
			binary[i,j] = 255
		else:
			binary[i,j] = 0
			
cv2.imwrite("Gray_Image_exp2.jpg",gray)
cv2.imwrite("Binary_Image_exp2.jpg",binary)
#gray_img = 0.114*b + 0.587*g + 0.299*r

