import cv2
import os

#jpeg
img = cv2.imread("dataset/campus2.jpeg")

# qualities = [90, 70, 50, 30, 10]

# print("=============FILE Quality COMPARISON TABLE=================")
# print("-----------------------------------------------------------")

# print("Quality  |  size in bytes   |   size in kb")
# print("------------------------------------------")

# for q in qualities:

#     cv2.imwrite(f'dataset/campus2{q}.jpeg' , img,[int(cv2.IMWRITE_JPEG_QUALITY),q])

#     size_bytes = os.path.getsize(f'dataset/campus2{q}.jpeg')
#     size_kb = size_bytes / 1024.0

#     print(f"{q}   |   {size_bytes}   |   {size_kb}")

#png
qualities = [0,3,5,7,9]

print("=============FILE Quality COMPARISON TABLE=================")
print("-----------------------------------------------------------")

print("Quality  |  size in bytes   |   size in kb")
print("------------------------------------------")

for q in qualities : 
    cv2.imwrite(f'dataset/campus2.png{q}', img , int[cv2.IMWRITE_PNG_COMPRESSION ,q])

    size_bytes = os.path.getsize(f'dataset/campus2{q}.png')
    size_kb = size_bytes / 1024.0
    
    print(f"{q}   |   {size_bytes}   |   {size_kb}")

