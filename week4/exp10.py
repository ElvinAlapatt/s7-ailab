import cv2
import numpy as np

img_path = '/mnt/d/ailab/week4/car.webp'
img = cv2.imread(img_path)
print("Img read successfully..\n")

gamma = float(input("Now Enter the gamma value : "))


table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype('uint8')

corrected = cv2.LUT(img,table)

cv2.imwrite("outputexp10.jpeg",corrected)



























