import cv2

img = cv2.imread("dataset/campus.jpeg")

gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray , 127 , 255 , cv2.THRESH_BINARY)


grayinv = 255 - gray 
bininv = 255 - binary
img_inv = 255 - img 

cv2.imshow("OG",img)
cv2.imshow("grayinv",grayinv)
cv2.imshow("bininv",bininv)
cv2.imshow("inv",img_inv)
cv2.waitKey(0)
cv2.destroyAllWindows()