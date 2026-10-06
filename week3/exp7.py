import cv2
import numpy as np

img_path = "/mnt/d/ailab/week3/car.webp"
img = cv2.imread(img_path)
h , w , _ = img.shape

print(f"Image height and width in pixels are {h}px X {w}px ")


r1 = int(input(f"Enter pixel (row: 0-{h-1}): "))
c1 = int(input(f"Enter pixel (col: 0-{w-1}): "))

r2 = int(input(f"Enter pixel (row: 0-{h-1}): "))
c2 = int(input(f"Enter pixel (col: 0-{w-1}): "))

ed , md = 0 , 0

dr = abs(r1-r2)
dc = abs(c1-c2)

md = dr + dc

ed = ((dr**2) + (dc**2))**0.5

print(f"Manhattan Distance is : {md} and Euclidean Distance is : {ed}")
