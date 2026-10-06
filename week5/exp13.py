import cv2 
import os 

img_path = 'week5\car.webp'
img = cv2.imread(img_path)

qualities = [90, 70, 50, 30, 10]

print('==================File Quality comparison Table==================\n')
print('-----------------------------------------------------------------\n')

print("Quality  |  File Size in bytes  |  File size in kb")
print('--------------------------------------------------')

for q in qualities:
    file_name = f"week5\output_for_q{q}.jpeg"

    cv2.imwrite(file_name,img,[int(cv2.IMWRITE_JPEG_QUALITY),q])

    size_bytes = os.path.getsize(file_name)
    size_kb = size_bytes / 1024.0

    print(f"{q}  |  {size_bytes}  |  {size_kb}")