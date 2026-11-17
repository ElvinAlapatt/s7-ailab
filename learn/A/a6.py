'''
A6. Convert the image to HSV and Grayscale. Show all three side by side in a 1x3 subplot with titles.
'''


import cv2
import matplotlib.pyplot as plt 


img_path = "dataset/campus2.jpeg"

img = cv2.imread(img_path)

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10,5))
plt.subplot(1,3,1)
plt.title('img')
plt.imshow(img)
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Gray")
plt.imshow(gray , cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title("HSV")
plt.imshow(img_hsv)
plt.axis('off')

plt.tight_layout()
plt.show()