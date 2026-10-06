import cv2
import os 

img_path = 'week5\img.png'
img = cv2.imread(img_path)

compression_levels = [0, 3, 5, 7, 9]

print("=================Compression Table=====================\n")
print("-------------------------------------------------------\n")

print(f'Compression Level  |  Size in bytes  |  Size in KB\n')

for lvl in compression_levels:
    file_name = f"week5\output_for_lvl{lvl}.png"

    cv2.imwrite(file_name,img,[int(cv2.IMWRITE_PNG_COMPRESSION),lvl])

    size_in_bytes = os.path.getsize(file_name)
    size_in_KB = size_in_bytes / 1024.0
    print(f"{lvl}  |  {size_in_bytes}  |  {size_in_KB}")