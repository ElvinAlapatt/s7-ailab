# CBL401 AI Lab: 2-Day Survival Guide and Question Bank

**Your situation:** 24 experiments look like a lot, but they collapse into about 8 building blocks. The professor can only "mix" things from these blocks. If you can write each block from memory, any mix is just two blocks glued together.

**The rule for these 2 days:** do not re-read the record. Type code yourself, run it, break it, fix it. Reading feels productive, typing is what the exam tests.

---

## PART 1: THE PLAN

### Tonight (Sunday), 20 minutes only, then sleep
1. Install and check: `pip install opencv-python numpy matplotlib pandas scikit-learn`
2. Put 3 images in a folder (a photo, a small logo PNG, a second photo). Make one small CSV (the weather dataset from Exp 20 or any 30-row CSV).
3. Read the "Core 10 Lines" section below once. Sleep. A tired brain does not retain syntax.

### Monday: Image Processing Day (about 8 hours of work in 90-min blocks)

| Block | Time | What | Goal |
|---|---|---|---|
| 1 | 9:00 to 10:30 | **Foundations**: read, show, shape, slicing/ROI, color spaces, resize, crop, flip, draw shapes/text, `copyMakeBorder` | Do Drills A1 to A8 |
| 2 | 10:45 to 12:15 | **Geometric + arithmetic**: translate, scale, rotate, add/subtract/blend, bitwise ops, masks | Do Drills B1 to B8 |
| 3 | 13:30 to 15:00 | **Point transforms + histograms**: negative, log, gamma, contrast stretch, threshold, histogram, equalization (manual and OpenCV) | Do Drills C1 to C8 |
| 4 | 15:15 to 16:45 | **Filters**: average, Gaussian, median, Laplacian, Sobel, Canny, sharpen, custom `filter2D` kernel | Do Drills D1 to D8 |
| 5 | 17:00 to 18:00 | **Compression + from-scratch (no OpenCV)**: JPEG/PNG/WEBP quality vs size, neighbors, distances, manual grayscale | Do Drills E1 to E6 |
| 6 | 19:00 to 20:30 | **Image-only mixed questions** (Part 3, Section F) | Do 4 of them with no notes |

Sleep by 23:00.

### Tuesday: ML Day + Mixing Day

| Block | Time | What | Goal |
|---|---|---|---|
| 1 | 9:00 to 10:30 | **Classification**: Naive Bayes (scratch + sklearn), KNN (scratch + sklearn), train/test split, confusion matrix, precision/recall/F1/accuracy | Drills G1 to G8 |
| 2 | 10:45 to 12:15 | **Clustering**: K-Means, GMM/EM, plotting clusters, centroids, choosing K | Drills H1 to H6 |
| 3 | 13:30 to 15:30 | **IMAGE + ML mixing** (this is where he will hit you): color quantization with K-Means, KNN on image features, histogram-based classification, segmentation by clustering | Part 3, Section I |
| 4 | 15:45 to 17:30 | **Cold run**: set a timer for 45 minutes, pick 2 random questions from different sections, solve with no notes | Repeat twice |
| 5 | 18:00 to 19:00 | **Viva prep**: answer the "Why" questions in Part 4 out loud | |
| 6 | 19:00 onward | Light review of your own mistakes list. **Sleep by 22:30.** | |

### Wednesday morning
Re-type the "Core 10 Lines" from memory and the logo-overlay solution. Eat something. Go.

### How to practice
- After every drill, write one line in a **mistakes list** (e.g. "forgot `cv2.resize` takes (width, height)"). That list is your last-night revision.
- If you are stuck for more than 10 minutes, read the hint, then close it and type it again from scratch.
- In the exam, if a library function exists use it. The professor wants the core idea, so also be ready to say what the function does internally.

---

## PART 2: CORE CHEAT SHEET (the stuff you must know cold)

### Core 10 Lines
```python
import cv2, numpy as np, matplotlib.pyplot as plt
img  = cv2.imread("a.jpg")                         # BGR, shape (h, w, 3), dtype uint8
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)       # shape (h, w)
rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        # for matplotlib
h, w = img.shape[:2]
plt.imshow(rgb); plt.axis("off"); plt.show()       # color: use rgb. gray: cmap="gray"
cv2.imwrite("out.jpg", img)
roi = img[y1:y2, x1:x2]                            # ROWS first (y), then columns (x)
small = cv2.resize(img, (new_w, new_h))            # resize takes (width, height) NOT (h, w)
```

### The 10 mistakes that kill marks
1. `imshow` in OpenCV (BGR) vs Matplotlib (RGB). Convert before `plt.imshow`.
2. `resize` takes `(w, h)`. `img.shape` gives `(h, w, c)`.
3. Slicing is `img[y1:y2, x1:x2]`. y first.
4. uint8 overflow: `200 + 100 = 44` in NumPy uint8. `cv2.add` saturates to 255. Use `cv2.add`/`cv2.subtract` or cast to int first.
5. ROI and logo must be the **same size** before `addWeighted`.
6. `cv2.imread` returns `None` on a bad path. Always check `if img is None`.
7. Laplacian/Sobel: use `cv2.CV_64F`, then `cv2.convertScaleAbs`. Using uint8 clips negative values.
8. Median/Gaussian kernel size must be **odd**.
9. For `cv2.imwrite` quality: JPEG/WEBP use quality (higher = better quality). PNG uses compression level 0 to 9 (higher = smaller file, same quality, lossless).
10. Alpha/opacity: `result = alpha*A + (1-alpha)*B`. `addWeighted(A, a, B, 1-a, 0)`.

### Quick reference (function: what it does)
| Task | Code |
|---|---|
| Grayscale (manual) | `0.299*R + 0.587*G + 0.114*B` |
| Binary | `cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)` (Otsu: add `+cv2.THRESH_OTSU`, threshold arg 0) |
| Adaptive threshold | `cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)` |
| Translate | `M=np.float32([[1,0,tx],[0,1,ty]]); cv2.warpAffine(img,M,(w,h))` |
| Rotate | `M=cv2.getRotationMatrix2D((w//2,h//2),angle,scale); cv2.warpAffine(img,M,(w,h))` |
| Flip | `cv2.flip(img, 0/1/-1)` (0 vertical, 1 horizontal, -1 both) |
| Crop | `img[y1:y2, x1:x2]` |
| Add/blend | `cv2.addWeighted(a, 0.7, b, 0.3, 0)` |
| Subtract / abs diff | `cv2.subtract(a,b)` / `cv2.absdiff(a,b)` |
| Logic ops | `cv2.bitwise_and/or/xor/not(a, b)`; with mask: `cv2.bitwise_and(img,img,mask=m)` |
| Negative | `255 - img` |
| Log | `c = 255/np.log(1+img.max()); s = c*np.log1p(img.astype(np.float32))` |
| Gamma | `s = 255*(r/255)**gamma` (gamma<1 brightens, gamma>1 darkens) |
| Contrast stretch | `(img - img.min())/(img.max()-img.min())*255` |
| Histogram | `np.bincount(gray.ravel(), minlength=256)` or `cv2.calcHist` |
| Equalize | `cv2.equalizeHist(gray)`; color: equalize only V of HSV (or Y of YCrCb) |
| Average blur | `cv2.blur(img,(k,k))` or `filter2D` with `np.ones((k,k))/k**2` |
| Gaussian | `cv2.GaussianBlur(img,(k,k),0)` |
| Median | `cv2.medianBlur(img,k)` (best for salt-and-pepper noise) |
| Laplacian | `cv2.Laplacian(gray,cv2.CV_64F,ksize=3)` |
| Sobel | `cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)` (x), `(0,1)` (y) |
| Canny | `cv2.Canny(gray,100,200)` |
| Sharpen | `sharp = img - lap` (Laplacian sharpening) or kernel `[[0,-1,0],[-1,5,-1],[0,-1,0]]` |
| Border | `cv2.copyMakeBorder(img,t,b,l,r,cv2.BORDER_CONSTANT,value=(0,0,0))` |
| Draw | `cv2.rectangle(img,(x1,y1),(x2,y2),(B,G,R),thickness)`, `cv2.circle(img,(cx,cy),r,color,-1)`, `cv2.putText` |
| Compression | JPEG `[cv2.IMWRITE_JPEG_QUALITY,q]`, PNG `[cv2.IMWRITE_PNG_COMPRESSION,c]`, WEBP `[cv2.IMWRITE_WEBP_QUALITY,q]`, size via `os.path.getsize` |
| MSE / PSNR | `mse=np.mean((a.astype(float)-b.astype(float))**2); psnr=10*np.log10(255**2/mse)` |

### ML templates
```python
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

df = pd.read_csv("data.csv")
X = df.iloc[:, :-1]; y = df.iloc[:, -1]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)

# KNN
from sklearn.neighbors import KNeighborsClassifier
m = KNeighborsClassifier(n_neighbors=5).fit(Xtr, ytr); p = m.predict(Xte)

# Naive Bayes (numeric features)   -> GaussianNB; (word counts) -> MultinomialNB
from sklearn.naive_bayes import GaussianNB
m = GaussianNB().fit(Xtr, ytr); p = m.predict(Xte)

# Metrics
print(confusion_matrix(yte, p), accuracy_score(yte, p))
print(precision_score(yte, p, average="weighted"), recall_score(yte, p, average="weighted"), f1_score(yte, p, average="weighted"))

# K-Means / GMM (unsupervised: no y)
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, n_init=10, random_state=42); labels = km.fit_predict(X); centers = km.cluster_centers_
from sklearn.mixture import GaussianMixture
gm = GaussianMixture(n_components=3, random_state=42).fit(X); labels = gm.predict(X); centers = gm.means_
```

### From-scratch ideas to be able to explain
- **KNN:** distance from the test point to every training point, sort, take K smallest, majority vote.
- **Naive Bayes:** `P(class) * product of P(feature | class)`, pick the highest. Laplace smoothing avoids zero probabilities. "Naive" = assumes features are independent.
- **K-Means:** pick K centroids, assign each point to the nearest, recompute centroids as the mean, repeat until stable. Hard assignment.
- **GMM/EM:** E-step computes the probability each point belongs to each Gaussian, M-step updates means, covariances and weights. Soft assignment.
- **Histogram equalization:** histogram, normalize to PDF, cumulative sum (CDF), map `new = round(CDF * 255)`.

### Image to ML bridge (the likely "mixing" pattern)
```python
# pixels -> feature table: each pixel is a row with 3 features (B,G,R)
pixels = img.reshape(-1, 3).astype(np.float32)          # (h*w, 3)
# ... fit KMeans on pixels, then paint each pixel with its centroid:
km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(pixels)
quant = km.cluster_centers_[km.labels_].reshape(img.shape).astype(np.uint8)
```
Another bridge: turn each image into a feature vector (mean intensity, 256-bin histogram, flattened small resize like 8x8) and feed those to KNN or Naive Bayes.

---

## PART 3: QUESTION BANK

Format: a question is *done* when it runs, shows the output, and you can explain each line. Hints are given only for the tricky ones.

### Section A: Foundations
- **A1.** Read an image, print its shape, dtype, min/max pixel value, and display it using both OpenCV and Matplotlib. Explain why colors look wrong if you skip the BGR to RGB conversion.
- **A2.** Split the image into B, G, R channels. Display each channel in grayscale. Then build an image where only the red channel is kept (others zero).
- **A3.** Crop the central 50% of the image and save it. Then paste that crop into the top-left corner of another image.
- **A4.** Draw a rectangle, a circle and the text "MEC" on the image. Save it.
- **A5.** Add a 20px white border around an image, then a 5px red border inside it (two borders of different widths and colors).
- **A6.** Convert the image to HSV and Grayscale. Show all three side by side in a 1x3 subplot with titles.
- **A7.** Resize an image to half using three interpolations (nearest, linear, cubic). Show them and explain the difference.
- **A8.** Flip horizontally, vertically and both. Display in a 2x2 grid with the original.

### Section B: Geometric, arithmetic, logical
- **B1.** Translate the image 80px right and 40px down. Make the translated image fully visible (canvas big enough, no cropping). *Hint:* output size `(w+80, h+40)`.
- **B2.** Rotate by 45 degrees **without cutting corners.** *Hint:* compute new bounding size `new_w = h*sin + w*cos`, shift the matrix: `M[0,2] += (new_w/2 - cx)`, `M[1,2] += (new_h/2 - cy)`.
- **B3.** Scale by 0.5 and by 2, show sizes with `shape`. Then scale back and compute MSE vs the original.
- **B4.** Blend two images with alpha = 0.2, 0.5, 0.8 and display all three. Then show that `cv2.add` and `a+b` give different results (explain the overflow).
- **B5.** Subtract two images and show the difference. Use `absdiff`, then threshold the difference to highlight "changed" pixels.
- **B6.** Create a white square and a white circle (300x300). Show AND, OR, XOR, NOT, and (square AND NOT circle).
- **B7.** Make a circular mask on a photo so everything outside the circle turns black. *Hint:* zeros mask, `cv2.circle(mask,center,r,255,-1)`, `bitwise_and(img,img,mask=mask)`.
- **B8.** **(Internal-exam question, learn this one fully)** Download a campus photo and a college logo. Place the logo in the **top-right corner at 50% opacity**, add a **3px border around the logo**, and a **10px border around the full image**.

**Full solution to B8:**
```python
import cv2
img  = cv2.imread("campus.jpg")
logo = cv2.imread("logo.png")
logo = cv2.resize(logo, (150, 150))                       # size of logo

# 3px border around logo (borders ADD size, so do this before placing)
logo = cv2.copyMakeBorder(logo, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(0, 0, 0))
lh, lw = logo.shape[:2]

H, W = img.shape[:2]
y1, x1 = 0, W - lw                                         # top-right corner
roi = img[y1:y1+lh, x1:x1+lw]
img[y1:y1+lh, x1:x1+lw] = cv2.addWeighted(logo, 0.5, roi, 0.5, 0)   # 50% opacity

# 10px border around whole image (do this LAST so it doesn't shift coordinates)
img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
cv2.imwrite("result.jpg", img)
```
*Variations he might ask:* bottom-left corner (`y1=H-lh, x1=0`), center (`y1=(H-lh)//2, x1=(W-lw)//2`), opacity 30%, remove the logo's white background first (threshold + mask), tiled watermark across the whole image, border color/width changes, border drawn **inside** the image using `cv2.rectangle` instead of adding pixels.

### Section C: Point transforms and histograms
- **C1.** Negative of a gray, binary and color image. Explain `s = L-1-r`.
- **C2.** Apply log transform to a dark image. Show before/after plus both histograms. Why does it brighten dark regions?
- **C3.** Apply gamma = 0.4, 1, 2.5 to a grayscale image **using a lookup table** (`cv2.LUT`). Compare with applying it directly.
- **C4.** Contrast stretching (min-max) on a low-contrast image. Print the min/max before and after.
- **C5.** Plot the histogram of a grayscale image **manually using loops** (no `calcHist`). Verify against `np.bincount`.
- **C6.** Implement histogram equalization **from scratch** (PDF, CDF, mapping) and compare with `cv2.equalizeHist` (they should match closely). Show before/after images and histograms.
- **C7.** Equalize a **color** image correctly (HSV V-channel). Then do it wrongly (equalize B, G, R separately) and explain why colors shift.
- **C8.** Threshold an image with fixed 127, Otsu, and adaptive. Show all three. When does each fail?

### Section D: Filters
- **D1.** Apply average filters 3x3, 5x5, 7x7 using **your own kernel with `filter2D`**. Then write the 3x3 average **using nested loops** (no OpenCV filter) and compare.
- **D2.** Add salt-and-pepper noise (set ~5% random pixels to 0 or 255). Denoise with average, Gaussian, median. Compute PSNR vs the clean image for each. Which wins and why?
- **D3.** Add Gaussian noise (`np.random.normal`), denoise with the same three filters, compare PSNR.
- **D4.** Laplacian (3x3, 5x5) and Sobel x, y, magnitude `sqrt(gx²+gy²)`. Show all.
- **D5.** Sharpen an image two ways: (a) `img - lap`, (b) unsharp mask `img + k*(img - blur)`.
- **D6.** Canny edges with two threshold pairs. Explain what the two thresholds do (hysteresis).
- **D7.** Apply a custom kernel you define: emboss `[[-2,-1,0],[-1,1,1],[0,1,2]]`. Display.
- **D8.** Edge-preserving blur using `cv2.bilateralFilter` vs Gaussian on the same image. What is different near edges?

### Section E: Compression and "no library" tasks
- **E1.** Save as JPEG at quality 90, 50, 10. Print a table (quality, KB) and compute PSNR of each vs the original. Show the trend in a plot.
- **E2.** Save as PNG with compression 0, 5, 9. Show that the file size changes but **pixels are identical** (`np.array_equal`). Explain lossless vs lossy.
- **E3.** Compare JPEG, PNG, WEBP for the same image at similar quality in one table sorted by size.
- **E4.** Convert RGB to grayscale and binary **without any OpenCV conversion functions** (use NumPy or loops). Compare with `cv2.cvtColor` using `np.abs` difference.
- **E5.** For a binary image, count the 4-neighbors and 8-neighbors of pixel (r, c) that have the same value. Then use that to find the count of pixels that are isolated (zero same-value neighbors).
- **E6.** Read two pixel coordinates from the user, print Euclidean, Manhattan and chessboard (D8) distance. Then compute the **distance transform** of a binary image with `cv2.distanceTransform`.

### Section F: Image-only mixed questions
- **F1.** Read an image, convert to grayscale, equalize, then apply Gaussian blur, then Canny, and display all stages in a 2x3 grid.
- **F2.** Add salt-and-pepper noise, remove it with a median filter, compute PSNR before and after, then save the cleaned image as JPEG at quality 30 and report the file size.
- **F3.** Rotate an image by 30 degrees, then scale to 70%, then translate by (50, 20), all combined into **one** transformation matrix. *Hint:* multiply 3x3 matrices, pass the top 2 rows to `warpAffine`.
- **F4.** Add a text watermark "MEC" in the bottom-right corner with 40% opacity, then apply a 5px black border, then negative of the result.
- **F5.** Take two images, blend them 60/40, convert the blend to HSV, equalize V, convert back, and save as PNG compression 9.
- **F6.** Given a binary image, find the number of white objects (connected components) with `cv2.connectedComponents`, then draw a bounding box around each one with `cv2.connectedComponentsWithStats`.
- **F7.** Apply log transform, then gamma, then show how the histogram changes after each step (3 histograms in one figure).
- **F8.** Make a mask for the brightest 10% of pixels (percentile threshold), apply the mask to the color image, and show the result.

### Section G: Classification (ML)
- **G1.** Implement Naive Bayes from scratch on a small categorical CSV with Laplace smoothing. Predict one new sample and print the probability for each class.
- **G2.** Run `GaussianNB` on Iris with a 70/30 split. Print confusion matrix, precision, recall, F1, accuracy.
- **G3.** KNN from scratch (Euclidean distance + majority vote) on Iris. Compare accuracy for K = 1, 3, 5, 7, 9 and **plot K vs accuracy**.
- **G4.** Same as G3 but with **Manhattan** distance. Compare with Euclidean. Why can they differ?
- **G5.** KNN **with feature scaling** (`StandardScaler` or manual min-max) vs without. Show accuracy for both. When does scaling matter most?
- **G6.** Text classification with `CountVectorizer` + `MultinomialNB` on a small CSV of sentences with labels. Use an 80/20 split, report all metrics, and predict a sentence typed by the user.
- **G7.** Do 5-fold cross-validation for KNN (K=5) on Iris using a manual loop (no `cross_val_score`). Report mean and std of accuracy.
- **G8.** From a confusion matrix given as a NumPy array, compute per-class precision and recall **manually**, then macro and weighted averages. Check against sklearn.

### Section H: Clustering
- **H1.** K-Means on a 2D CSV for K = 2, 3, 4. Plot before and after, centroids as red X marks. Print inertia for each K.
- **H2.** **Elbow method:** plot K = 1 to 8 vs inertia. State which K you would choose and why.
- **H3.** Implement K-Means **from scratch** (random init, assign, update, stop when centroids stop moving). Compare centroids with sklearn's.
- **H4.** GMM for K = 2, 3, 4. Print weights, means and covariances. Plot with different colors.
- **H5.** Compare K-Means and GMM on an elongated (stretched) cluster dataset (use `make_blobs` and a linear transform). Explain why GMM handles it better.
- **H6.** Print the silhouette score for K = 2 to 6 and pick the best K.

### Section I: IMAGE + ML (the hardest, most likely mixing zone)
- **I1.** **Color quantization:** reduce a photo to K = 2, 4, 8, 16 colors using K-Means on pixels. Show all results and file sizes when saved as PNG. *Hint:* see "Image to ML bridge".
- **I2.** **Segmentation:** run K-Means (K=3) on pixels of an image and display each cluster as a separate binary mask.
- **I3.** **Same as I1 but with GMM.** Compare visually and explain soft vs hard assignment.
- **I4.** **Background removal (basic):** cluster pixels into 2 groups, treat the cluster with the larger area as background, make it white.
- **I5.** **Feature extraction + KNN:** you have 20 small images (or generate 20 synthetic ones: 10 bright, 10 dark). Use mean intensity and standard deviation as features, train KNN (K=3), test on a new image.
- **I6.** **Histogram features + Naive Bayes:** extract a 16-bin histogram from each grayscale image as a feature vector, then classify bright vs dark images with `GaussianNB`.
- **I7.** **Flattened pixels + KNN:** resize images to 8x8, flatten to 64 features, classify. Use `sklearn.datasets.load_digits` as a ready dataset to practice, then show 5 correctly and 5 wrongly classified digits.
- **I8.** **Preprocessing effect:** train KNN on `load_digits` raw, then after Gaussian blur, then after thresholding. Compare accuracy and comment.
- **I9.** **K-Means on image + edge detection:** quantize to K colors, run Canny on the quantized image, and compare edges with Canny on the original.
- **I10.** **Image data augmentation + classifier:** augment each training image with a rotation and a flip (doubling the data), retrain KNN, and compare accuracy before and after augmentation.
- **I11.** **Clustering by color for similarity:** compute a mean color (B, G, R) of 10 images, cluster them with K-Means (K=2), and print which images fall in which cluster.
- **I12.** **Compress then classify:** save digits as JPEG at quality 10, reload, classify with KNN. Compare accuracy vs lossless. Explain how lossy compression affects ML.

### Section J: "Surprise" variants (the professor style)
These combine unusual constraints. Practice by reading each one slowly and listing which blocks are involved.
- **J1.** Read an image, rotate by 90 degrees without using `cv2.rotate` or `warpAffine` (use NumPy: `np.rot90` or transpose + flip). Explain the index mapping.
- **J2.** Create an image from scratch: 400x400 gradient from black to white, then draw a green circle and a blue border, then save as JPEG 50 and PNG, comparing sizes.
- **J3.** Take an image, produce a 3-level quantization manually (pixel < 85 becomes 0, < 170 becomes 127, else 255), without `cv2.threshold`.
- **J4.** Write a function `apply_pipeline(img, steps)` where `steps` is a list like `["gray", "blur", "edge"]` that applies the steps in order.
- **J5.** Given an image folder, read all images, resize to 100x100, compute mean B, G, R for each, store in a DataFrame, and save as CSV. Then cluster it.
- **J6.** Crop only the top-right quarter, apply a different filter to each of the 4 quadrants (average, Gaussian, median, Laplacian), and stitch them back with `np.hstack` / `np.vstack`.
- **J7.** Detect whether an image is **dark, normal or bright** from its histogram (mean intensity thresholds), and apply gamma automatically to correct it.
- **J8.** Make a "before / after" side-by-side comparison automatically for any function you write, with PSNR and file size printed under each image.

---

## PART 4: VIVA / "CORE IDEA" QUESTIONS (he loves these)

Answer each in 2 to 3 sentences out loud.

1. Why is OpenCV BGR and not RGB? How do you fix display issues?
2. What is a pixel? What does `uint8` limit? What happens on overflow?
3. Difference between grayscale and binary images? Why the weights 0.299, 0.587, 0.114?
4. Difference between interpolation methods (nearest, bilinear, bicubic)?
5. Why does `addWeighted` need weights summing to about 1?
6. What is a convolution/kernel? Why odd sizes? What happens at the border pixels?
7. Average vs Gaussian vs Median filter: which removes salt-and-pepper noise best, and why?
8. What does the Laplacian detect? Why does it amplify noise?
9. What does a histogram show? What does a narrow histogram mean? What does equalization do and why can it over-enhance?
10. Log vs gamma: which brightens dark regions, and what does gamma < 1 do?
11. Lossy vs lossless compression. Why does PNG's compression level not change quality?
12. Why does JPEG show blocks at low quality? (8x8 DCT blocks, quantization)
13. Why must you scale features before KNN?
14. K too small vs too large in KNN?
15. What does "naive" mean in Naive Bayes? Why Laplace smoothing?
16. Precision vs recall vs F1. When do you care about each?
17. K-Means vs GMM: hard vs soft clustering, spherical vs elliptical.
18. How do you choose K in K-Means? (elbow, silhouette)
19. What is EM? Describe the E-step and M-step.
20. How would you use K-Means on an image? What do the cluster centers represent? (the dominant colors)
21. Why is accuracy misleading on imbalanced data?
22. What is overfitting? How does the train/test split help detect it?

---

## PART 5: EXAM-DAY TACTICS

1. **Read the question twice and underline the blocks.** Example: "equalize then cluster" = histogram + KMeans.
2. **Build and test in small steps.** Run after every 3 or 4 lines. Print `img.shape` constantly.
3. **Get partial output first**, even a wrong-looking one. A running program with a half answer beats a perfect program that never ran.
4. **Use library functions but know the internals.** If he asks "what is inside?", answer with the idea from the cheat sheet.
5. **If the dataset is missing:** make one. `from sklearn.datasets import load_iris, load_digits, make_blobs` give data instantly. For images, create one with NumPy (`np.random.randint(0,256,(200,200,3),dtype=np.uint8)`).
6. **If a function name slips:** type `cv2.` then Tab, or `help(cv2.func)`. In Colab, Shift+Tab shows the signature.
7. **Say what you are doing out loud** as you code. He values the core idea over syntax.

---

## YOUR HONEST EXPECTATION

You will not memorize everything in 2 days and you do not need to. Your target: **be able to produce a working solution for any single block in under 10 minutes, and glue two blocks together in under 25.** Sections A to E build the blocks; F to J train the gluing. If you only have time to be great at one thing, it is Section B8 (logo overlay), Section I1 (K-Means color quantization), Section G3 (KNN) and Section C6 (histogram equalization from scratch). Those four cover most of what he can twist.