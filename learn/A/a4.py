'''
A4. Draw a rectangle, a circle and the text "MEC" on the image. Save it.
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt 

img = np.zeros((300,600),dtype='uint8')

cv2.rectangle(img,(int(0.25*600),int(0.25*300)),(int(0.75*600),int(0.75*300)),200,10)

cv2.circle(img,(img.shape[1]//2,img.shape[0]//2),100,200,10)

cv2.putText(img , "MEC", ((img.shape[1]//2)-30 ,(img.shape[0]//2)+5), cv2.FONT_HERSHEY_SIMPLEX, 1.0 , 255 , 2)

cv2.imshow("rect",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#for saving it uncomment from lines 22
#cv2.imwrite("output.jpeg",img)
#print("Image saved as output.jpeg")