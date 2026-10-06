import cv2
import matplotlib.pyplot as plt

img_path = 'week4\car.webp'

img = cv2.imread(img_path , cv2.IMREAD_GRAYSCALE)

equalized_img = cv2.equalizeHist(img)

# now we have to plot the original image and the original histogram and then equalized img and the equalized historgam
plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(img , cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2,2,2)
plt.hist(img.ravel(),bins=256, range=[0,256],color='gray')
plt.title("Original Histogram")
plt.xlim([0,256])

plt.subplot(2,2,3)
plt.imshow(equalized_img , cmap='gray')
plt.title('Eq Image')
plt.axis('off')

plt.subplot(2,2,4)
plt.hist(equalized_img.ravel(),bins=256, range=[0,256],color='gray')
plt.title("Eq Histogram")
plt.xlim([0,256])

plt.tight_layout()
plt.show()