import cv2
import numpy as np

img_path = '/mnt/d/ailab/week3/img.jpeg'

img = cv2.imread(img_path)
print("Img read and the shape of the image is ", img.shape)

gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
binary_img = cv2.threshold(gray_img,127,255,cv2.THRESH_BINARY)

h , w , _ = img.shape

n4_count = np.zeros((h,w),dtype='unit8')
n8_count = np.zeros((h,w),dtype='unit8')

orthogonal_neighbours = [(-1,0),(1,0),(0,1),(0,-1)]
diagonal_neighbours = [(-1,-1),(1,-1),(1,1),(-1,1)]

for r in range(h):
	for c in range(w):
		curr_color = binary_img[r,c]
		
		#c4
		c4 = 0
		for dr , dc in orthogonal_neighbours:
			nr = r + dr
			nc = c + dc 
			if 0 <= nr < h and 0 <= nc < width:
				if binary_img[nr,nc] == curr_color:
					c4 += 1
			n4_count[r,c] = c4
		#c8
		
		




































'''
import cv2
import numpy as np

def compute_pixel_neighbors(image_path=None):
    # ---------------------------------------------------------
    # STEP 1: Load or create a binary image
    # ---------------------------------------------------------
    if image_path:
        # Load grayscale image from disk
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Threshold: > 127 becomes 255 (White/BG), else 0 (Black/FG)
        binary_img = np.where(gray >= 128, 255, 0).astype(np.uint8)
    else:
        # Small 5x5 test image if no file is provided
        # 0 = Black (FG), 255 = White (BG)
        binary_img = np.array([
            [255, 255,   0,   0,   0],
            [255, 255,   0, 255,   0],
            [  0,   0,   0, 255, 255],
            [  0, 255,   0,   0, 255],
            [  0,   0,   0, 255, 255]
        ], dtype=np.uint8)

    height, width = binary_img.shape

    # ---------------------------------------------------------
    # STEP 2: Create empty matrices to store counts
    # ---------------------------------------------------------
    n4_result = np.zeros((height, width), dtype=np.uint8)
    n8_result = np.zeros((height, width), dtype=np.uint8)

    # Relative coordinates: (row_change, col_change)
    orthogonal_neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]      # Top, Bottom, Left, Right
    diagonal_neighbors   = [(-1, -1), (-1, 1), (1, -1), (1, 1)]    # 4 Corners

    # ---------------------------------------------------------
    # STEP 3: Iterate through EVERY pixel (r, c)
    # ---------------------------------------------------------
    for r in range(height):
        for c in range(width):
            current_color = binary_img[r, c]

            # --- Count 4-Neighbours ---
            count_4 = 0
            for dr, dc in orthogonal_neighbors:
                nr = r + dr
                nc = c + dc
                # Boundary check: make sure neighbor is inside image
                if 0 <= nr < height and 0 <= nc < width:
                    if binary_img[nr, nc] == current_color:
                        count_4 += 1

            # --- Count Diagonal Neighbours ---
            count_diag = 0
            for dr, dc in diagonal_neighbors:
                nr = r + dr
                nc = c + dc
                # Boundary check
                if 0 <= nr < height and 0 <= nc < width:
                    if binary_img[nr, nc] == current_color:
                        count_diag += 1

            # --- Store Results ---
            n4_result[r, c] = count_4
            n8_result[r, c] = count_4 + count_diag  # 8-neighbours = 4-cross + 4-diagonals

    return binary_img, n4_result, n8_result


# -------------------------------------------------------------
# RUN AND DISPLAY
# -------------------------------------------------------------
if __name__ == "__main__":
    # Run test on the small sample grid
    img, n4, n8 = compute_pixel_neighbors()

    print("=== 1. Original Binary Image ===")
    print(img)

    print("\n=== 2. Number of 4-Neighbours (Same Color) ===")
    print(n4)

    print("\n=== 3. Number of 8-Neighbours (Same Color) ===")
    print(n8)

    # If you want to run it on your own image file in the lab:
    # img, n4, n8 = compute_pixel_neighbors("your_image.png")
    # cv2.imwrite("n4_output.png", (n4 * (255 // 4)).astype(np.uint8))
    # cv2.imwrite("n8_output.png", (n8 * (255 // 8)).astype(np.uint8))
    
'''
