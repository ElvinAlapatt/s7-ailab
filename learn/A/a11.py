"""
A9. Brighten an image by +60 using NumPy (wrong, overflows) and using cv2.add or clipping (right). Show both and explain.
"""

import cv2
import numpy as np

img_path = "dataset/campus2.jpeg"

img = cv2.imread(img_path)

img2 = cv2.add(img,60)

cv2.imshow("img",img)
cv2.imshow("img2",img2)
cv2.waitKey(0)
cv2.destroyAllWindows()