import cv2
import matplotlib.pyplot as plt

img_path = 'week6\car.webp'
img = cv2.imread(img_path)

img_rgb = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)

blur3x3 = cv2.medianBlur(img_rgb,3)
blur5x5 = cv2.medianBlur(img_rgb,5)
blur7x7 = cv2.medianBlur(img_rgb,7)

plt.figure(figsize=(12,5))

plt.subplot(1,3,1)
plt.imshow(blur3x3)
plt.title("3x3 median blur")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(blur5x5)
plt.title("5x5 median blur")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(blur7x7)
plt.title("7x7 median blur")
plt.axis('off')

plt.tight_layout()
plt.show()