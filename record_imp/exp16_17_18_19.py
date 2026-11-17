#spatial averaging blur
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("dataset/campus2.jpeg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

blur3 = cv2.blur(img, (3, 3))
blur5 = cv2.blur(img, (5, 5))
blur7 = cv2.blur(img, (7, 7))

plt.subplot(2, 2, 1)
plt.imshow(img)
# plt.title("Original")

plt.subplot(2, 2, 2)
plt.imshow(blur3)
# plt.title("3x3")

plt.subplot(2, 2, 3)
plt.imshow(blur5)
# plt.title("5x5")

plt.subplot(2, 2, 4)
plt.imshow(blur7)
# plt.title("7x7")

plt.show()


#gaussian blur
blur3 = cv2.GaussianBlur(img, (3, 3), 0)
blur5 = cv2.GaussianBlur(img, (5, 5), 0)
blur7 = cv2.GaussianBlur(img, (7, 7), 0)

cv2.imshow("Original", img)
cv2.imshow("Gaussian 3x3", blur3)
cv2.imshow("Gaussian 5x5", blur5)
cv2.imshow("Gaussian 7x7", blur7)

cv2.waitKey(0)
cv2.destroyAllWindows()

#median blur
blur3 = cv2.medianBlur(img, 3)
blur5 = cv2.medianBlur(img, 5)
blur7 = cv2.medianBlur(img, 7)

cv2.imshow("Original", img)
cv2.imshow("Median 3x3", blur3)
cv2.imshow("Median 5x5", blur5)
cv2.imshow("Median 7x7", blur7)

cv2.waitKey(0)
cv2.destroyAllWindows()


#laplacian 
gray = cv2.imread("dataset/campus.jpeg", 0)

lap3 = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F, ksize=3))
lap5 = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F, ksize=5))
lap7 = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F, ksize=7))

cv2.imshow("Original", gray)
cv2.imshow("Laplacian 3x3", lap3)
cv2.imshow("Laplacian 5x5", lap5)
cv2.imshow("Laplacian 7x7", lap7)

cv2.waitKey(0)
cv2.destroyAllWindows()