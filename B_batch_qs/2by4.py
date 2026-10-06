import cv2
import numpy as np

img = cv2.imread("/mnt/d/ailab/B_batch_qs/car.webp")

H , W , _ = img.shape

h_step = H // 2
w_step = W // 4

count = 0
'''
for r in range(2):
	current_row = []
	for c in range(4):
		y1 , y2 = r * h_step , (r+1)*h_step
		x1 , x2 = c * w_step , (c+1)*w_step
		
		if r == 1:
			y2 = H
		if c == 3:
			x2 == W
			
		count += 1
		block = img[y1:y2 , x1:x2]
		cv2.imwrite(f"block_image{count}.jpg",block)
'''	
# now this works to make it better

row_strips = []

for r in range(2):
	current_row = []
	for c in range(4):
		y1 , y2 = r * h_step , (r+1)*h_step
		x1 , x2 = c * w_step , (c+1)*w_step
		
		if r == 1:
			y2 = H
		if c == 3:
			x2 == W
			
		count += 1
		block = img[y1:y2 , x1:x2]
		block_resized = cv2.resize(block , (h_step,w_step))
		
		cv2.rectangle(block_resized, (0, 0), (w_step - 1, h_step - 1), (255, 255, 255), 1)
		
		current_row.append(block_resized)
		
	row_strips.append(cv2.hconcat(current_row))
	
final_strips = cv2.vconcat(row_strips)

cv2.imwrite("Grid.jpg",final_strips)
		
		
