import cv2 
import numpy as np

img_path = "/mnt/d/ailab/B_batch_qs/car.webp"

img = cv2.imread(img_path)

h , w , d = img.shape

output = np.zeros((h,w,d),dtype='uint8')

color1 = input("Enter the RGB values with a seperation of space (color to be replaced) : ")
rgb1=[int(x) for x in color1.split()]
bgr1=rgb1[::-1]

color2 = input("Enter the RGB values with a seperation of space :(color you want) : ")
rgb2=[int(x) for x in color2.split()]
bgr2=rgb2[::-1]

for i in range(h):
	for j in range(w):
		b , g , r = img[i,j]
		if bgr1[0]-50 <= b <= bgr1[0]+50:
			b = bgr2[0]
		if bgr1[1]-50 <= g <= bgr1[1]+50:
			g = bgr2[1]
		if bgr1[2]-50 <= r <= bgr1[2]+50:
			r = bgr2[2]
		output[i,j] = (b,g,r)
		
cv2.imwrite("outputcolorreplace.jpeg",output)
