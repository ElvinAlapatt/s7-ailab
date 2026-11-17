"""
A8. Flip horizontally, vertically, both, using slicing (no cv2.flip). Show in a 2x2 grid.

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt 

img_path = "dataset/campus2.jpeg"

img =cv2.imread(img_path)

h = cv2.flip(img , 1) # horizontal
v = cv2.flip(img , 0) #vertical
b = cv2.flip(img , -1) #both

# img = img[::,::-1] # this flips it horizontal
# img = img[::-1,::] # this flips  vertical
# img = img[::-1,::-1]

img = cv2.cvtColor( img , cv2.COLOR_BGR2RGB)
h = cv2.cvtColor( h , cv2.COLOR_BGR2RGB) 
v = cv2.cvtColor( v , cv2.COLOR_BGR2RGB) 
b = cv2.cvtColor( b , cv2.COLOR_BGR2RGB) 

plt.figure(figsize=(10,6))

plt.subplot(2,2,1)
plt.title("OG")
plt.imshow(img)
plt.axis('off')

plt.subplot(2,2,2)
plt.title("Horizontal")
plt.imshow(h)
plt.axis('off')

plt.subplot(2,2,3)
plt.title("Vertical")
plt.imshow(v)
plt.axis('off')

plt.subplot(2,2,4)
plt.title("BOTH")
plt.imshow(b)
plt.axis('off')

plt.tight_layout()
plt.show()