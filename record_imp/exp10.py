import cv2
import numpy as np

img = cv2.imread("dataset/campus2.jpeg")

gamma = float(input("Enter Gamma : "))

table = np.array([((i/255.0)**gamma)*255 for i in range(256)])

corrected = cv2.LUT(img,table)

cv2.imshow("Gamma corrected", corrected)
cv2.waitKey(0)
cv2.destroyAllWindows()