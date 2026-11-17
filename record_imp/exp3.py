"""
translation , scaling , rotation
"""
import numpy as np
import cv2

img = cv2.imread('dataset/campus2.jpeg')

#scaling -> resize
#we can specify exact dimensions like this
imgr = cv2.resize(img , (400,300) , interpolation=cv2.INTER_LINEAR)
#we can scale by factor of (x,y)
imgfr = cv2.resize(img , None ,fx=1.5 , fy=1.5 , interpolation=cv2.INTER_LINEAR)

# cv2.imshow("img",img)
# cv2.imshow("imgr",imgr)
# cv2.imshow("imgfr",imgfr)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#Translation -> shifting
#we use a translation matrix

trans_m = np.float32([[1,0,100],[0,1,50]])
imgtrans=cv2.warpAffine(img , trans_m , (img.shape[1],img.shape[0]))
# cv2.imshow("imgtrans",imgtrans)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#Rotation
#it uses a rotation matrix which can be called by helper function in cv2
#then use warpAffine
#rotation needs a point(center , or any point) angle and then a scale 
center = (img.shape[1]//2,img.shape[0]//2)
angle = 45
scale = 1

rot_matrix = cv2.getRotationMatrix2D(center,angle,scale)
rotated_img = cv2.warpAffine(img,rot_matrix,(img.shape[1],img.shape[0]))
# cv2.imshow("imgrot",rotated_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()