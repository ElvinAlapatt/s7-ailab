import cv2
import matplotlib.pyplot as plt 
import numpy as np

img = cv2.imread("dataset/campus2.jpeg")
imgM = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)


eq = cv2.equalizeHist(img)

plt.figure(figsize=(10,6))

plt.subplot(2,2,1)
plt.imshow(imgM)
plt.title("original")
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(eq , cmap='gray')
plt.title("equalized")
plt.axis('off')

plt.subplot(2,2,2)
plt.hist(img.ravel() , 256 , [0,256])
plt.title("original")

plt.subplot(2,2,4)
plt.hist(eq.ravel() , 256 , [0,256])
plt.title("original")

plt.tight_layout()
plt.show()
