"""
A9. Rotate by 90 degrees using NumPy only. Explain the index mapping (i, j) -> (j, h-1-i).
"""

import cv2
import numpy as np

img_path = "dataset/campus2.jpeg"

img = cv2.imread(img_path)

h , w , _ = img.shape
# 3 ways  to implement this one using numpy rot function and other using tthe index mapping

trans = img.transpose(1, 0, 2)
trans_rot = img.transpose(1, 0, 2)[::,::-1]

#rot 
#this fucntion uses counter clockwise so to rotate 90 degree you have to specify -1
rot_img = np.rot90(img,-1) 


rotated_img = np.zeros((w,h,3),dtype='uint8')

for i in range(h):
    for j in range(w):
        rotated_img[j,h-i-1] = img[i,j]

cv2.imshow("rot using numpy",rot_img)
cv2.imshow("rotated img",rotated_img)
cv2.imshow("trans img",trans )
cv2.imshow("trans_rot img",trans_rot )
cv2.waitKey(0)
cv2.destroyAllWindows()