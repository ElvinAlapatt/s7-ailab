"""
A3. Crop the central 50% of the image and save it. Then paste that crop into the top-left corner of another image.
"""

import cv2
import numpy as np
import matplotlib as plt

img_path  = "dataset/campus2.jpeg"

img = cv2.imread(img_path)
another_img = cv2.imread("dataset/campus.jpeg")

h , w , _ = img.shape 


start_y , end_y = int(h*0.25) , int(h*0.75)
start_x , end_x = int(w*0.25) , int(w*0.75)

img_cropped = img[start_y:end_y , start_x:end_x]

cv2.imshow("img",img)
cv2.imshow("img crop",img_cropped)


h2 ,w2 , _ = img_cropped.shape


print("img_cropped", img_cropped.shape)
print("anotherimg ",another_img.shape)
# for i in range(h2):
#     for j in range(w2):
#         another_img[i,j] = img_cropped[i,j]

another_img[2:h2+2 , 2:w2+2 ] = img_cropped

cv2.imshow("new image", another_img)

cv2.waitKey(0)
cv2.destroyAllWindows()