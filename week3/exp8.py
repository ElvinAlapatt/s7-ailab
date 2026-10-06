import cv2
import numpy as np

img_path = "/mnt/d/ailab/week3/car.webp" 

img = cv2.imread(img_path)

h , w , _ = img.shape

output = np.zeros((h,w,3),dtype='uint8')

for i in range(h):
	for j in range(w):
		output[i,j] = 255 - img[i,j]

cv2.imwrite("output.jpeg",output)
