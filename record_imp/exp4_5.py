import cv2 
import numpy as np

#exp4
#addition and subtraction
img1 = cv2.imread('dataset/campus.jpeg')
img2 = cv2.imread('dataset/campus2.jpeg')
img2 = cv2.resize(img2 , (img1.shape[1],img1.shape[0]), interpolation=cv2.INTER_AREA)

added = cv2.addWeighted(img1 , 0.5 , img2 , 0.5 , 0)

subtracted = cv2.subtract(img1,img2)

# cv2.imshow("added",added)
# cv2.imshow("subtracted",subtracted)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#exp5
#logical operations
r = np.zeros((500,500),dtype='uint8')
c = np.zeros((500,500),dtype='uint8')

# rec = cv2.copyMakeBorder(r , 20 , 20 , 20 , 20, cv2.BORDER_CONSTANT,value=255)
rec = cv2.rectangle(r,(0,0),(500,500),255,40)
cir = cv2.circle(c , (250,250),245,255,-1)
cir = cv2.add(c,cir)

_, rec = cv2.threshold(rec , 127 , 255 , cv2.THRESH_BINARY_INV)

#operations
orimg = cv2.bitwise_or(rec,cir)
andimg = cv2.bitwise_and(rec,cir)
xorimg = cv2.bitwise_xor(rec,cir)

cv2.imshow("rec",rec)
cv2.imshow("cir",cir)
cv2.imshow("or",orimg)
cv2.imshow("and",andimg)
cv2.imshow("xor",xorimg)
cv2.waitKey(0)
cv2.destroyAllWindows()