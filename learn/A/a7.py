"""
A7. Resize an image to half using three interpolations (nearest, linear, cubic). Show them and explain the difference.
"""

import cv2
import numpy as np

img_path = "dataset/campus2.jpeg"

img =cv2.imread(img_path)

h , w , _ = img.shape

n = cv2.resize(img , (int(h*0.5),int(w*0.5)) , interpolation=cv2.INTER_NEAREST)
l = cv2.resize(img , (int(h*0.5),int(w*0.5)) , interpolation=cv2.INTER_LINEAR)
c = cv2.resize(img , (int(h*0.5),int(w*0.5)) , interpolation=cv2.INTER_CUBIC)

cv2.imshow("img",img)
cv2.imshow("nearest",n)
cv2.imshow("linear",l)
cv2.imshow("cubic",c)
cv2.waitKey(0)
cv2.destroyAllWindows()


#doubts
# what's the difference between these 3 and what's a spline?