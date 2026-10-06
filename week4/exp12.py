import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = 'week4\car.webp'

img = cv2.imread(img_path)

#bgr to rgb
img_rgb = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)

#rgb to hsv
img_hsv = cv2.cvtColor(img , cv2.COLOR_RGB2HSV)

#unpack hue saturation and value h s v 
h , s ,v = cv2.split(img_hsv)

#now only equalise v
v_eq = cv2.equalizeHist(v)

#merge it 
hsv_eq = cv2.merge([h,s,v_eq])

#then convert it into rgb 
img_eq_rgb = cv2.cvtColor(hsv_eq,cv2.COLOR_HSV2RGB)

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis('off')

plt.subplot(2,2,2)
plt.hist(img_rgb.ravel(), bins=256 , range=[0,256] , color='gray', alpha=0.8)
plt.title('Original Histogram')
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(img_eq_rgb)
plt.title("Equalized only v")
plt.axis('off')

plt.subplot(2,2,4)
plt.hist(img_eq_rgb.ravel(), bins=256 , range=[0,256] , color='gray', alpha=0.8)
plt.title('Equalized Histogram')
plt.axis('off')

plt.tight_layout()
plt.show()