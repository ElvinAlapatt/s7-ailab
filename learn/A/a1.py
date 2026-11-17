'''
question
Read an image, print its shape, dtype, min/max pixel value, and display it using both OpenCV and Matplotlib. 
Explain why colors look wrong if you skip the BGR to RGB conversion.
'''

import numpy as np 
import cv2
import matplotlib.pyplot as plt 

img_path = "dataset/campus.jpeg"

img = cv2.imread(img_path)
if img is None:
    print("Image is not opening") 

print("shape = ",img.shape)

print("dtype = ",img.dtype)

print("min pixel value : ", img.min())

print("max pixel value : ", img.max())

cv2.imshow("opencv window",img)

imgformatplotlib = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(imgformatplotlib)
plt.title("matplotlib window")
plt.axis('off')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
