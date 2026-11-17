"""
A5. Add a 20px white border around an image, then a 5px red border inside it (two borders of different widths and colors).
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

img_path = "dataset/campus2.jpeg"

img = cv2.imread(img_path)

h,w,_ = img.shape

#this is not the best way to do it 
# cv2.rectangle(img,(0,0),(w-1,h-1),(0,0,0),20)
# cv2.rectangle(img,(10,10),(w-10,h-10),(0,0,255),5)


img_b = cv2.copyMakeBorder(img , top = 20 , bottom = 20 , left = 20 , right = 20 , borderType=cv2.BORDER_CONSTANT,value=(0,0,0))
cv2.rectangle(img_b,(20,20),(w+20,h+20),(0,0,255),5)

# cv2.imshow("image",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

imgmat = cv2.cvtColor(img_b , cv2.COLOR_BGR2RGB)


plt.figure()
plt.imshow(imgmat)
plt.axis('off')
plt.title("img")
plt.show()