'''
import cv2

#i have to read an image 
#i have to convert it into single color channel grayscale
#i have to convert it into binary 
# i have to write the grayscale and binary image to the memory 

img_path = "/mnt/d/ailab/week1/exp1_sample_img.jpg"
img = cv2.imread(img_path)
cv2.imshow("OG IMAGE",img)
cv2.waitKey(0)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("GRAY IMG",gray_img)
cv2.imwrite(gray_img)

ret_val , binary_img = cv2.threshold(gray_img,127,255,cv2.THRESH_BINARY)
cv2.imshow("BINARY IMG",binary_img)
cv2.imwrite(binary_img)

'''


#--------------------------THE ABOVE CODE IS ENOUGH without imshow but use matplotlib pyplot just in case-------------------------

import cv2
import os
import matplotlib.pyplot as plt


#i have to read an image 
#i have to convert it into single color channel grayscale
#i have to convert it into binary 
# i have to write the grayscale and binary image to the memory 

img_path = "/mnt/d/ailab/week1/exp1_sample_img.jpg"
img = cv2.imread(img_path)
cv2.imshow("OG IMAGE",img)
cv2.waitKey(0)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("GRAY IMG",gray_img)
cv2.imwrite(gray_img)

ret_val , binary_img = cv2.threshold(gray_img,127,255,cv2.THRESH_BINARY)
cv2.imshow("BINARY IMG",binary_img)
cv2.imwrite(binary_img)

 # Create a side-by-side plot layout using Matplotlib
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original Image
axes[0].imshow(img_rgb)
axes[0].set_title("Original Image")
axes[0].axis('off')
        
# Grayscale Image (cmap='gray' forces it to render in single-channel gray)
axes[1].imshow(gray_img, cmap='gray')
axes[1].set_title("Grayscale Image")
axes[1].axis('off')
        
# Binary Image
axes[2].imshow(binary_img, cmap='gray')        
axes[2].set_title("Binary Image")
axes[2].axis('off')
