"""
Histogram equalization of color image
we use hsv
"""

import cv2
import matplotlib.pyplot as plt 

img = cv2.imread("dataset/campus2.jpeg")

imghsv = cv2.cvtColor(img , cv2.COLOR_BGR2HSV)
# print(imghsv.shape)

h , s , v = cv2.split(imghsv)
# print(v.shape)

v_eq = cv2.equalizeHist(v)

eq = cv2.merge([h,s,v_eq])

img_eq_rgb = cv2.cvtColor(eq,cv2.COLOR_HSV2RGB)

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(img_eq_rgb)
plt.title("Original")
plt.axis('off')

plt.subplot(2,2,2)
plt.hist(img.ravel(), bins=256 , range=[0,256] , color='blue', alpha=0.8)
plt.title('Original Histogram')
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(img_eq_rgb)
plt.title("Equalized only v")
plt.axis('off')

plt.subplot(2,2,4)
plt.hist(img_eq_rgb.ravel(), bins=256 , range=[0,256] , color='blue', alpha=0.8)
plt.title('Equalized Histogram')
plt.axis('off')

plt.tight_layout()
plt.show()