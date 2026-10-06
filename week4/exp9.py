import cv2
import numpy as np

img_path = "/mnt/d/ailab/week4/car.webp" 

img = cv2.imread(img_path , cv2.IMREAD_GRAYSCALE)
h , w = img.shape

output = np.zeros((h,w),dtype='uint8')

# s = c * (log(1+r)) where r is the pixel intensity , c is the scaling factor and s is the output pixel intensity

# c = 255/(log(1+max(r)))

max_r = float(np.max(img))

if max_r == 0:
	c = 0.0
else:
	c = 255/(np.log(1+max_r))


for i in range(h):
	for j in range(w):
		r = img[i,j]
		
		
		s = c * (np.log(1+r))
		
		output[i,j] = s

cv2.imwrite('inputexp9.jpeg',img)
cv2.imwrite("outputexp9.jpeg",output)
