"""
A10. Split the image into 4 quadrants with slicing, convert only the top-left to grayscale, only bottom-right to negative, stitch back with np.hstack/np.vstack.
"""

import cv2
import numpy as np

img = cv2.imread("dataset/campus2.jpeg")

h , w , _ = img.shape

q1 = img[:h//2:,:w//2:]
q2 = img[h//2::,:w//2:]
q3 = img[:h//2:,w//2::]
q4 = img[h//2::,w//2::]


q1 = cv2.cvtColor(q1 , cv2.COLOR_BGR2GRAY)
q4 = cv2.cvtColor(q4 , cv2.COLOR_BGR2GRAY)
_ , q4 = cv2.threshold(q4 , 127 , 255 , cv2.THRESH_BINARY)

q1 = cv2.cvtColor(q1 , cv2.COLOR_GRAY2BGR)
q4 = cv2.cvtColor(q4 , cv2.COLOR_GRAY2BGR)

q1_q2 = np.vstack((q1,q2))
q3_q4 = np.vstack((q3,q4))
full = np.hstack((q1_q2 , q3_q4))

cv2.imshow("original",img)
cv2.imshow("q1",q1)
cv2.imshow("q2",q2)
cv2.imshow("q3",q3)
cv2.imshow("q4",q4)
cv2.imshow("full",full)
cv2.waitKey(0)
cv2.destroyAllWindows()