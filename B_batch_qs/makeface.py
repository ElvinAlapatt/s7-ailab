import cv2
import numpy as np


img = cv2.imread('/mnt/d/ailab/B_batch_qs/makefaceimg.jpg')
h, w = img.shape[:2]

binary_img = np.zeros((h,w),dtype='uint8')

left_eye = (int(w*0.25),int(h*0.25),int(w*0.40),int(h*0.40))
 
right_eye = (int(w*0.55),int(h*0.25),int(w*0.70),int(h*0.40))


cv2.rectangle(binary_img , (left_eye[0],left_eye[1]), (left_eye[2],left_eye[3]), 255, -1)

cv2.rectangle(binary_img , (right_eye[0],right_eye[1]), (right_eye[2],right_eye[3]), 255, -1)

mouth_center = (int(w*0.50),int(h*0.75))
mouth_radius = int(w*0.20)

cv2.circle(binary_img,mouth_center,mouth_radius,255,-1)

#cv2.imshow("Binary",binary_img)

gray1ch = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
gray3ch = cv2.cvtColor(gray1ch , cv2.COLOR_GRAY2BGR)

binary_inv = cv2.bitwise_not(binary_img)

color_part = cv2.bitwise_and(img,img,mask = binary_inv)
gray_part = cv2.bitwise_and(gray3ch , gray3ch, mask=binary_img)

final_result = cv2.add(color_part,gray_part)

cv2.imwrite("makefacefinal.jpg",final_result)

cv2.waitKey(0)
cv2.destroyAllWindows()
