# CBL401 AI Lab: Final Survival Guide v2 (Exam: tomorrow 10:30 AM)

**What changed:** Naive Bayes and EM/GMM are OUT (Exps 20, 21, 22). Only **K-Means (Exp 23)** and **KNN (Exp 24)** from ML. Everything image processing (Exps 1 to 19) is IN.

**What Sir told you, translated into work:**
| His hint | What it means for you |
|---|---|
| Understand image array manipulation | An image is just a NumPy array. Slicing, masks, channels, `reshape`, dtype. (Part 1 and Part 3A) |
| Learn to do filtering **without the library** | Write convolution with loops/NumPy yourself: average, Gaussian, median, Laplacian, Sobel. (Part 2 is the most important page in this file) |
| VIVA per lab syllabus, may include numericals | Part 5 has 14 hand-calculation problems with answers |
| Questions similar to internal lab exam | Logo overlay + borders (B8). Do it first, then do variations |

**Priority order (do the bank in this order):** Image processing, then K-Means, then KNN, then mixed questions, then numericals and viva.

**Rule:** do not read the solutions first. Try, get stuck, peek, close, retype. Question marked **(star)** = must do. If you run out of energy, the stars alone are enough to pass.

---

## PART 1: CORE CHEAT SHEET (know this cold)

### The 8 lines that start every program
```python
import cv2, numpy as np, matplotlib.pyplot as plt
img  = cv2.imread("a.jpg")                         # BGR, shape (h, w, 3), dtype uint8
if img is None: raise FileNotFoundError("bad path")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)       # shape (h, w)
rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        # matplotlib wants RGB
h, w = img.shape[:2]
plt.imshow(rgb); plt.axis("off"); plt.show()       # gray: plt.imshow(gray, cmap="gray")
cv2.imwrite("out.jpg", img)
```
Colab: use `from google.colab.patches import cv2_imshow` then `cv2_imshow(img)` (your record uses this).

### The mistakes that cost marks
1. OpenCV is **BGR**; matplotlib is **RGB**. Convert before `plt.imshow`.
2. `cv2.resize(img, (width, height))`, but `img.shape` is `(height, width, channels)`.
3. Slicing is `img[y1:y2, x1:x2]`. Rows (y) first.
4. uint8 overflow: NumPy `200 + 100 = 44` (wraps). `cv2.add(200,100) = 255` (saturates). `50 - 100`: NumPy gives 206, `cv2.subtract` gives 0.
5. For `addWeighted`, both images must be the **same size**.
6. Kernel sizes for Gaussian and median must be **odd**.
7. Laplacian/Sobel: use `cv2.CV_64F` then `cv2.convertScaleAbs`, otherwise negatives get clipped.
8. When you do arithmetic on pixels in NumPy, cast first: `img.astype(np.float32)`. Cast back with `np.clip(x,0,255).astype(np.uint8)` at the end.
9. Borders **add** pixels (`copyMakeBorder`), so do the logo border before placing the logo, and the full-image border **last**.

### Function quick reference
| Task | Code |
|---|---|
| Grayscale (manual) | `(0.299*R + 0.587*G + 0.114*B)` where `B,G,R = img[:,:,0], img[:,:,1], img[:,:,2]` |
| Binary | `cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]` or manual `np.where(gray>127,255,0)` |
| Negative | `255 - img` |
| Log | `c = 255/np.log(256); s = c*np.log(1+r.astype(np.float32))` (matches your Exp 9) |
| Gamma | `s = 255*(r/255)**gamma` (gamma < 1 brightens, > 1 darkens) |
| Histogram | `np.bincount(gray.ravel(), minlength=256)` or `plt.hist(gray.ravel(),256,[0,256])` |
| Equalize | `cv2.equalizeHist(gray)`; color: equalize only the V channel of HSV |
| Translate | `M=np.float32([[1,0,tx],[0,1,ty]]); cv2.warpAffine(img,M,(w,h))` |
| Scale | `cv2.resize(img,None,fx=1.5,fy=1.5)` |
| Rotate | `M=cv2.getRotationMatrix2D((w//2,h//2),angle,1); cv2.warpAffine(img,M,(w,h))` |
| Add / subtract | `cv2.addWeighted(a,0.7,b,0.3,0)`, `cv2.subtract(a,b)`, `cv2.absdiff(a,b)` |
| Logic ops | `cv2.bitwise_and / or / xor / not` |
| Border | `cv2.copyMakeBorder(img,t,b,l,r,cv2.BORDER_CONSTANT,value=(B,G,R))` |
| Average filter | `cv2.filter2D(img,-1,np.ones((k,k),np.float32)/k**2)` |
| Gaussian | `cv2.GaussianBlur(img,(k,k),0)` |
| Median | `cv2.medianBlur(img,k)` |
| Laplacian | `cv2.Laplacian(gray,cv2.CV_64F)` |
| Sobel | `cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)` (x), `(0,1)` (y) |
| JPEG / WEBP / PNG save | `[cv2.IMWRITE_JPEG_QUALITY,q]`, `[cv2.IMWRITE_WEBP_QUALITY,q]`, `[cv2.IMWRITE_PNG_COMPRESSION,c]`; size: `os.path.getsize(f)/1024` |
| PSNR | `mse=np.mean((a.astype(float)-b.astype(float))**2); 10*np.log10(255**2/mse)` |
| Salt and pepper | `r=np.random.rand(*g.shape); n=g.copy(); n[r<0.025]=0; n[r>0.975]=255` |

---

## PART 2: FILTERING WITHOUT THE LIBRARY (his #1 hint, memorize this)

**The idea:** slide a small matrix (kernel) over the image. At each pixel, multiply the kernel with the neighbourhood and sum. Pad the borders so the output has the same size. (OpenCV's `filter2D` does correlation, which is exactly this loop. For symmetric kernels like average and Gaussian, correlation = convolution.)

### Template 1: general convolution (loops). Tested, matches `cv2.filter2D` to within 1 grey level.
```python
def conv2d(g, k):
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    p = np.pad(g.astype(np.float32), ((ph, ph), (pw, pw)), mode="constant")   # zero padding
    out = np.zeros(g.shape, np.float32)
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            out[i, j] = np.sum(p[i:i+kh, j:j+kw] * k)
    return out

# use it
k_avg = np.ones((3, 3), np.float32) / 9
res = np.clip(conv2d(gray, k_avg), 0, 255).astype(np.uint8)
```

### Template 2: the same thing but fast (no pixel loops). Use this if the image is big.
```python
def conv_fast(g, k):
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    p = np.pad(g.astype(np.float32), ((ph, ph), (pw, pw)))
    out = np.zeros(g.shape, np.float32)
    for dy in range(kh):
        for dx in range(kw):
            out += k[dy, dx] * p[dy:dy+g.shape[0], dx:dx+g.shape[1]]
    return out
```
Color image: run the function on each channel: `cv2.merge([f(c) for c in cv2.split(img)])`.

### Kernels to write from memory
```python
avg3     = np.ones((3,3), np.float32) / 9
laplace  = np.array([[0, 1, 0],[1,-4, 1],[0, 1, 0]], np.float32)
laplace8 = np.array([[1, 1, 1],[1,-8, 1],[1, 1, 1]], np.float32)
sobel_x  = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32)     # detects vertical edges
sobel_y  = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], np.float32)     # detects horizontal edges
sharpen  = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], np.float32)    # = identity + (-laplace)
emboss   = np.array([[-2,-1,0],[-1,1,1],[0,1,2]], np.float32)
```

### Gaussian kernel from the formula
```python
def gauss_kernel(ks, sigma):
    ax = np.arange(ks) - ks // 2
    xx, yy = np.meshgrid(ax, ax)
    g = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return (g / g.sum()).astype(np.float32)          # normalize so weights sum to 1
# gauss_kernel(3,1) = [[.075,.124,.075],[.124,.204,.124],[.075,.124,.075]]
```

### Median filter from scratch (not a kernel multiplication, it is a sort)
```python
def median_filter(g, ks=3):
    r = ks // 2
    p = np.pad(g, r, mode="edge")
    out = np.zeros_like(g)
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            out[i, j] = np.median(p[i:i+ks, j:j+ks])
    return out
```

### Sobel magnitude, Laplacian sharpening, unsharp mask (all built on conv2d)
```python
gx, gy = conv2d(gray, sobel_x), conv2d(gray, sobel_y)
mag = np.clip(np.sqrt(gx**2 + gy**2), 0, 255).astype(np.uint8)

lap = conv2d(gray, laplace)
sharp = np.clip(gray.astype(np.float32) - lap, 0, 255).astype(np.uint8)   # center is negative, so subtract

blur = conv2d(gray, gauss_kernel(5, 1))
unsharp = np.clip(gray + 1.5 * (gray - blur), 0, 255).astype(np.uint8)    # k=1.5
```

### Other "from scratch" building blocks
```python
# grayscale and binary with no cv2 conversion (your Exp 2)
b, g, r = img[:,:,0].astype(float), img[:,:,1].astype(float), img[:,:,2].astype(float)
gray_m   = (0.299*r + 0.587*g + 0.114*b).astype(np.uint8)
binary_m = np.where(gray_m > 127, 255, 0).astype(np.uint8)

# histogram + equalization from scratch
hist = np.bincount(gray.ravel(), minlength=256)
cdf  = hist.cumsum()
cdf_min = cdf[cdf > 0].min()
lut  = np.round((cdf - cdf_min) / (gray.size - cdf_min) * 255).astype(np.uint8)
eq   = lut[gray]                                   # matches cv2.equalizeHist exactly (tested)

# rotate 90 without cv2:  np.rot90(img)  or  img.transpose(1,0,2)[::-1]   (counter-clockwise)
# flip: img[:, ::-1] (horizontal), img[::-1] (vertical), img[::-1, ::-1] (both)

# 4 and 8 neighbours of pixel (r,c) with same value (your Exp 6)
def same_neighbors(a, r, c, eight=False):
    offs = [(-1,0),(1,0),(0,-1),(0,1)]
    if eight: offs += [(-1,-1),(-1,1),(1,-1),(1,1)]
    return sum(0 <= r+dr < a.shape[0] and 0 <= c+dc < a.shape[1] and a[r+dr,c+dc] == a[r,c] for dr,dc in offs)

# distances (your Exp 7)
dx, dy = abs(x2-x1), abs(y2-y1)
euclid, manhattan, chess = (dx**2+dy**2)**0.5, dx+dy, max(dx,dy)
```

### The logo overlay (internal exam question). Learn this one fully.
```python
img  = cv2.imread("campus.jpg")
logo = cv2.imread("logo.png")
logo = cv2.resize(logo, (150, 150))
logo = cv2.copyMakeBorder(logo, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(0, 0, 0))   # 3px border on logo
lh, lw = logo.shape[:2]
H, W = img.shape[:2]
roi = img[0:lh, W-lw:W]                                                              # top-right
img[0:lh, W-lw:W] = cv2.addWeighted(logo, 0.5, roi, 0.5, 0)                          # 50% opacity
img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255,255,255))  # 10px border LAST
cv2.imwrite("result.jpg", img)
```
Variations: bottom-left `img[H-lh:H, 0:lw]`, center `y=(H-lh)//2, x=(W-lw)//2`, other opacity (`alpha` for logo, `1-alpha` for roi), border drawn **inside** with `cv2.rectangle(img,(0,0),(W-1,H-1),color,thickness)` (no size change), remove logo's white background first.

---

## PART 3: ML ESSENTIALS (K-Means, then KNN)

### K-Means (Exp 23)
**Idea:** pick K centroids, assign every point to the nearest, move each centroid to the mean of its points, repeat until nothing moves.
```python
# Library version (your Exp 23)
import pandas as pd, matplotlib.pyplot as plt
from sklearn.cluster import KMeans
df = pd.read_csv("clustering_data.csv"); X = df.iloc[:, :2].values
fig, ax = plt.subplots(2, 2, figsize=(12, 10)); ax = ax.flatten()
ax[0].scatter(X[:,0], X[:,1], c="gray"); ax[0].set_title("Before clustering")
for i, k in enumerate([2, 3, 4], start=1):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X); C = km.cluster_centers_
    ax[i].scatter(X[:,0], X[:,1], c=labels, cmap="viridis")
    ax[i].scatter(C[:,0], C[:,1], c="red", marker="X", s=150)
    ax[i].set_title(f"K = {k}   inertia = {km.inertia_:.1f}")
plt.tight_layout(); plt.show()
```
```python
# From scratch (tested)
def kmeans(X, k, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)].astype(float)       # random initial centroids
    for _ in range(iters):
        d = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)    # (n, k) distances
        labels = d.argmin(axis=1)                                    # assign
        newC = np.array([X[labels == j].mean(axis=0) if np.any(labels == j) else C[j] for j in range(k)])
        if np.allclose(newC, C): break                               # converged
        C = newC
    return labels, C
```
**K-Means on an image (color quantization / segmentation):**
```python
pixels = img.reshape(-1, 3).astype(np.float32)            # every pixel becomes a row of 3 features (B,G,R)
km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(pixels)
quant = km.cluster_centers_[km.labels_].reshape(img.shape).astype(np.uint8)   # paint each pixel with its centroid color
```
Cluster centers = the dominant colors. `labels_.reshape(h, w)` = the segmentation map.

### KNN (Exp 24)
**Idea:** to classify a point, find the K closest training points and take the majority vote. No training phase (lazy learner).
```python
# From scratch (your Exp 24 core, shortened)
from collections import Counter
def knn_predict(Xtr, ytr, Xte, k):
    preds = []
    for t in Xte:
        d = np.sqrt(((Xtr - t) ** 2).sum(axis=1))      # Euclidean distance to all training points
        idx = np.argsort(d)[:k]                         # indices of K nearest
        preds.append(Counter(ytr[idx]).most_common(1)[0][0])   # majority vote
    return np.array(preds)

# Evaluation (copy from Exp 24)
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
cm  = confusion_matrix(y_test, y_pred, labels=np.unique(y))
acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
```
```python
# Library version + 70/30 split
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
for k in [3, 5, 7, 9]:
    print(k, KNeighborsClassifier(n_neighbors=k).fit(Xtr, ytr).score(Xte, yte))
```
Manual 70/30 split (your record does this): `idx=np.random.permutation(len(X)); n=int(0.7*len(X)); tr,te=idx[:n],idx[n:]`.

### Bridges between image and ML (where he will mix)
- **Pixels as data:** `img.reshape(-1,3)` gives one row per pixel. K-Means on this = segmentation or quantization.
- **KNN on pixels:** label a few pixels by region (sky = 0, grass = 1), train KNN on their (B,G,R), classify every pixel, paint the result.
- **Images as data:** each image becomes one row of features (mean intensity, 16-bin histogram, or `load_digits` where each image is 64 numbers). Then KNN.

---

## PART 4: THE QUESTION BANK

(star) = must do. Aim: do every star, then the rest as energy allows. Hints are in italics. Full code for the trickiest ones is in Part 4X right after.

### Section A: Image array manipulation (the "understand arrays" part)
- **A1 (star).** Read an image. Print shape, dtype, size, min, max, mean. Show it with matplotlib correctly. *BGR to RGB.*
- **A2 (star).** Split into B, G, R. Show each as grayscale. Build an image keeping only red (others zero). Then swap the R and B channels using slicing only (`img[:,:,::-1]`) and explain what you see.
- **A3 (star).** Crop the center 50% and save it. Paste that crop into the top-left corner of another image (same dtype, check it fits).
- **A4.** Draw a rectangle, circle and text "MEC" on an image using cv2 drawing functions. Then draw the same rectangle **using only array slicing** (set pixels to a color).
- **A5 (star).** Add a 20px white border, then a 5px red border outside it. Print shapes after each.
- **A6.** Flip horizontally, vertically, both, using slicing (no `cv2.flip`). Show in a 2x2 grid.
- **A7.** Rotate by 90 degrees using NumPy only. Explain the index mapping `(i, j) -> (j, h-1-i)`.
- **A8.** Make a 300x300 image from scratch: left half black, right half white, a gray square in the middle. Then a black-to-white gradient (`np.tile(np.arange(256), (256,1))`).
- **A9.** Brighten an image by +60 using NumPy (wrong, overflows) and using `cv2.add` or clipping (right). Show both and explain.
- **A10.** Split the image into 4 quadrants with slicing, convert only the top-left to grayscale, only bottom-right to negative, stitch back with `np.hstack/np.vstack`.

### Section B: Geometric, arithmetic, logical, point transforms
- **B1 (star).** Logo overlay: top-right, 50% opacity, 3px border on logo, 10px border on full image. (Solution in Part 2.)
- **B1b (star).** Same, but **bottom-left**, **30% opacity**, **5px red border on logo**, **15px black border on image**. Then save as JPEG quality 70.
- **B1c.** Tile the logo (small, 20% opacity) across the whole image as a watermark. *Loop over positions with step = logo size, clip at edges.*
- **B2 (star).** Translate by (100, 50), scale by 1.5, rotate 45 degrees. Show all three with shapes. (Exp 3.) Then make the **translated image not lose pixels** (canvas `(w+tx, h+ty)`).
- **B3.** Rotate 45 degrees without cutting corners. *New size `nw = h*sin + w*cos`, `nh = h*cos + w*sin`, then shift `M[0,2] += nw/2 - w/2`, `M[1,2] += nh/2 - h/2`.*
- **B4 (star).** Blend two images 0.7/0.3 and subtract them (resize first). (Exp 4.) Show why `a + b` differs from `cv2.add(a, b)`.
- **B5.** `absdiff` between two images, threshold it, show changed pixels.
- **B6 (star).** 300x300 white square and white circle. Show AND, OR, XOR, NOT. (Exp 5.)
- **B7.** Circular mask on a photo: outside becomes black. Then do the inverse (inside black).
- **B8 (star).** Negative of gray, binary, color. (Exp 8.)
- **B9 (star).** Log transform on a grayscale image, with the formula `c = 255/ln(256)`. Show before/after. Why does it brighten dark regions? (Exp 9.)
- **B10 (star).** Gamma correction for gamma = 0.4, 1.0, 2.5 (Exp 10). Do it with direct formula **and** with a lookup table `cv2.LUT`.
- **B11.** Contrast stretching (min-max). Print min/max before and after.
- **B12.** Fixed threshold 127 vs Otsu vs your own manual threshold (`np.where`). Compare.

### Section C: Histogram and compression
- **C1 (star).** Plot the histogram of a gray image by **counting with loops** and verify against `np.bincount`. (Exp 12 does this.)
- **C2 (star).** Histogram equalization with `cv2.equalizeHist`, showing images and histograms before/after. (Exp 11.)
- **C3 (star).** Histogram equalization **from scratch** (hist, cdf, LUT) and check it matches OpenCV.
- **C4 (star).** Equalize a **color** image via HSV V-channel (Exp 12). Then show what goes wrong if you equalize B, G, R separately.
- **C5 (star).** Save an image as JPEG at quality 90, 70, 50, 30, 10. Table of quality vs KB. (Exp 13.)
- **C6.** Save as PNG with compression 0, 3, 5, 7, 9. Show sizes change but pixels are identical (`np.array_equal`). (Exp 14.)
- **C7.** Save as WEBP at quality 100, 70, 50, 30, 10. (Exp 15.) Then one table comparing JPEG, PNG, WEBP sizes.
- **C8.** Compute PSNR of each JPEG quality against the original. Plot quality vs PSNR.

### Section D: Filtering (library AND from scratch). This section decides your marks.
- **D1 (star).** Averaging filters 3x3, 5x5, 7x7 with `cv2.filter2D` in a 2x2 plot. (Exp 16.)
- **D2 (star).** Write the 3x3 averaging filter with **nested loops, no OpenCV filter**. Compare with `cv2.filter2D` using `np.abs(a-b).max()`.
- **D3 (star).** Gaussian blur 3x3, 5x5, 7x7 using `cv2.GaussianBlur`. (Exp 17.) Then build the Gaussian kernel yourself and apply it with your `conv2d`.
- **D4 (star).** Median blur 3x3, 5x5. (Exp 18.) Then write **median from scratch**.
- **D5 (star).** Laplacian filter. (Exp 19.) Then apply the Laplacian kernel with your own `conv2d`, and sharpen with `img - lap`.
- **D6 (star).** Add salt-and-pepper noise (5%). Denoise using average, Gaussian, median. PSNR for each. Which wins and why? *Median: it ignores outliers instead of averaging them in.*
- **D7.** Add Gaussian noise (`np.random.normal(0, 25, shape)`), denoise with the same three. Now who wins? *Gaussian/average: noise is spread out, averaging cancels it.*
- **D8 (star).** Sobel x, y, and magnitude using **your own conv2d** with the Sobel kernels. Compare with `cv2.Sobel`.
- **D9.** Unsharp masking: `img + k*(img - blur)` for k = 0.5, 1, 2. Show over-sharpening.
- **D10.** Apply a custom 3x3 kernel typed by the user at runtime (emboss, or a user-entered matrix).
- **D11.** Show how the effect grows with kernel size: apply average 3, 7, 15. What happens to edges? To noise?
- **D12.** Apply the filter to a **color** image channel by channel with your own function.
- **D13.** Zero padding vs edge padding: apply your 3x3 average and show the border pixels differ. Which looks better and why?

### Section E: Neighbour and distance (Exps 6, 7)
- **E1 (star).** For a binary image (from a small NumPy array), compute the 4-neighbour and 8-neighbour same-value count for every pixel. Print both matrices. (Exp 6.)
- **E2.** Using E1, find how many pixels are isolated (zero same-value 8-neighbours).
- **E3 (star).** Read two pixel coordinates, print Euclidean, Manhattan, chessboard distance. (Exp 7.)
- **E4.** Make a distance map: for every pixel in a 50x50 grid, its Euclidean distance from (25,25). Show as an image.

### Section F: K-Means questions
- **F1 (star).** K-Means on a CSV (first two columns), K = 2, 3, 4, plot before and after with red X centroids. (Exp 23 exactly.)
- **F2 (star).** K-Means **from scratch** (assign, update, converge). Compare centroids with sklearn.
- **F3 (star).** **Color quantization:** reduce an image to K = 2, 4, 8, 16 colors with K-Means on pixels. Show all. Print number of unique colors in each (`np.unique(q.reshape(-1,3),axis=0)`).
- **F4 (star).** **Segmentation:** K-Means with K=3 on pixels. For each cluster show a binary mask and the masked color image (`cv2.bitwise_and(img,img,mask=m)`). Print pixel count in each cluster.
- **F5.** Elbow method: plot K = 1 to 8 vs `inertia_`. Pick K and justify.
- **F6.** Cluster using **grayscale intensity only** (1 feature): `gray.reshape(-1,1)`, K=2. Compare with Otsu threshold.
- **F7.** Cluster using color **plus position** (features `[B,G,R,x,y]`). What changes in the result? *Neighbouring pixels cluster together.*
- **F8.** Background removal: K=2 on pixels, make the larger cluster white.
- **F9.** Run K-Means on `load_iris().data[:, :2]` for K = 3 and compare labels with true labels using a confusion matrix. Why can cluster numbers be permuted?

### Section G: KNN questions
- **G1 (star).** KNN **from scratch** on Iris, 70/30 split, K = 3, 5, 7, 9, printing confusion matrix, precision, recall, accuracy, F1. (Exp 24 exactly.)
- **G2 (star).** Same using sklearn `KNeighborsClassifier`. Plot K (1 to 15) vs accuracy.
- **G3.** Manhattan distance instead of Euclidean. Compare accuracy.
- **G4.** With and without feature scaling (`(X-X.min(0))/(X.max(0)-X.min(0))`). When does scaling matter? *When one feature has a much larger range.*
- **G5 (star).** **KNN on pixels:** pick 50x50 sky region (class 0) and grass region (class 1) from a photo, train KNN on (B,G,R) and classify every pixel, paint class 0 blue and class 1 green.
- **G6 (star).** **KNN on images:** `load_digits`, 70/30, K=3. Report accuracy. Show 5 correctly and 5 wrongly classified digits.
- **G7.** KNN on synthetic images: generate 10 dark and 10 bright 32x32 images with NumPy, use (mean, std) as features, classify a new one.
- **G8.** Effect of preprocessing: KNN on `load_digits` raw vs after thresholding vs after blurring. Compare.
- **G9.** KNN regression idea: predict a missing pixel's brightness as the average of its K nearest known pixels. (Concept question, a small code is enough.)

### Section H: Mixed questions (the "he can combine anything" zone)
- **H1 (star).** Read image, grayscale, equalize, Gaussian blur, Canny. Show all stages in a 2x3 grid.
- **H2 (star).** Add salt-and-pepper noise, remove with **your own median filter**, print PSNR before and after, save the result as JPEG quality 30 and print the file size.
- **H3 (star).** Add the logo overlay (B1), then K-Means quantize the final image to 8 colors.
- **H4.** Convert to grayscale, take its negative, apply log transform, show 3 histograms in one figure.
- **H5.** Rotate by 30 degrees, then K-Means with K=3 on the rotated image. (Black corners become their own cluster. Explain.)
- **H6.** Blur with **your own** Gaussian filter, then K-Means (K=3) segmentation. Does blurring first give cleaner segments? Compare.
- **H7.** Take 10 images (or generated ones), compute mean (B,G,R) for each, cluster into 2 groups with K-Means, print which image is in which cluster.
- **H8.** Apply Sobel (your own) to get an edge map, threshold it, count edge pixels. Then count edge pixels again after Gaussian blur first. Explain the drop.
- **H9.** Quantize to K colors with K-Means, then save as PNG and JPEG. Compare file sizes against the original. Why does PNG shrink so much?
- **H10.** KNN on `load_digits` after adding salt-and-pepper noise to test images. Then after median-filtering them. Compare accuracy.
- **H11.** Write `pipeline(img, steps)` where `steps = ["gray","blur","edge"]` runs those in order.
- **H12.** Quadrant filtering: top-left average, top-right Gaussian, bottom-left median, bottom-right Laplacian, stitched into one image.

---

## PART 4X: KEY SOLUTIONS (tested)

### Pixel-wise KNN classification (G5)
```python
img = cv2.imread("a.jpg")
sky   = img[10:60, 10:60].reshape(-1, 3)         # choose boxes that really are sky / grass in YOUR image
grass = img[120:170, 10:60].reshape(-1, 3)
Xtr = np.vstack([sky, grass]); ytr = np.array([0]*len(sky) + [1]*len(grass))
knn = KNeighborsClassifier(n_neighbors=5).fit(Xtr, ytr)
lab = knn.predict(img.reshape(-1, 3)).reshape(img.shape[:2])
out = np.zeros_like(img); out[lab == 0] = (255, 0, 0); out[lab == 1] = (0, 255, 0)     # BGR: blue, green
```

### K-Means segmentation masks (F4)
```python
pixels = img.reshape(-1, 3).astype(np.float32)
km = KMeans(3, n_init=10, random_state=0).fit(pixels)
L = km.labels_.reshape(img.shape[:2])
for k in range(3):
    mask = (L == k).astype(np.uint8) * 255
    seg = cv2.bitwise_and(img, img, mask=mask)
    print(k, (L == k).sum(), "pixels")
```

### Salt-and-pepper noise + denoising + PSNR (D6, H2)
```python
def psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    return 99 if mse == 0 else 10 * np.log10(255**2 / mse)
r = np.random.rand(*gray.shape)
noisy = gray.copy(); noisy[r < 0.025] = 0; noisy[r > 0.975] = 255          # 5% corrupted
for name, f in [("avg", cv2.blur(noisy,(3,3))), ("gauss", cv2.GaussianBlur(noisy,(3,3),0)), ("median", cv2.medianBlur(noisy,3))]:
    print(name, round(psnr(gray, f), 2))
```

### Rotate without cropping (B3)
```python
h, w = img.shape[:2]
M = cv2.getRotationMatrix2D((w/2, h/2), 45, 1)
c, s = abs(M[0,0]), abs(M[0,1])
nw, nh = int(h*s + w*c), int(h*c + w*s)
M[0,2] += nw/2 - w/2; M[1,2] += nh/2 - h/2
rot = cv2.warpAffine(img, M, (nw, nh))
```

### Compression table (C5)
```python
import os, pandas as pd
rows = []
for q in [90, 70, 50, 30, 10]:
    f = f"q{q}.jpg"; cv2.imwrite(f, img, [cv2.IMWRITE_JPEG_QUALITY, q])
    rows.append({"quality": q, "KB": round(os.path.getsize(f)/1024, 2), "PSNR": round(psnr(img, cv2.imread(f)), 2)})
print(pd.DataFrame(rows))
```

### Gamma with lookup table (B10)
```python
for g in [0.4, 1.0, 2.5]:
    lut = np.array([255 * (i/255) ** g for i in range(256)], np.uint8)
    out = cv2.LUT(gray, lut)                  # same as 255*(gray/255)**g but 256 computations instead of h*w
```

### Elbow + silhouette (F5)
```python
inertias = [KMeans(k, n_init=10, random_state=0).fit(X).inertia_ for k in range(1, 9)]
plt.plot(range(1, 9), inertias, "o-"); plt.xlabel("K"); plt.ylabel("inertia"); plt.show()
```

---

## PART 5: NUMERICALS (hand calculation, he said these may come in viva). Answers at the end of each.

**N1. Filters on a patch.** Patch = `[[10,20,30],[40,50,60],[70,80,90]]`. Find the output at the centre for: (a) 3x3 average, (b) 3x3 median, (c) Laplacian `[[0,1,0],[1,-4,1],[0,1,0]]`, (d) Sobel-x, (e) Sobel-y, (f) Sobel magnitude.
*Answers:* (a) 450/9 = **50**. (b) sorted middle = **50**. (c) 20+40+60+80 - 4(50) = **0** (a linear ramp has no second derivative). (d) (-10+30) + 2(-40+60) + (-70+90) = **80**. (e) (70+160+90) - (10+40+30) = **240**. (f) sqrt(80² + 240²) = **252.98**.

**N2. Histogram equalization.** 3-bit image (L = 8, levels 0 to 7), 16 pixels. Counts for levels 0..7: `2, 4, 3, 2, 2, 1, 1, 1`. Find the new level for each using `s = round(7 * CDF / 16)`.
*Answers:* CDF = 2, 6, 9, 11, 13, 14, 15, 16. New levels: 0 to **1**, 1 to **3**, 2 to **4**, 3 to **5**, 4 to **6**, 5 to **6**, 6 to **7**, 7 to **7**.

**N3. Distances.** P = (2, 3), Q = (7, 15). Euclidean, city-block, chessboard?
*Answers:* dx = 5, dy = 12. Euclidean **13**, Manhattan **17**, chessboard **12**.

**N4. Storage.** 1024 x 768 image. Size in bytes as: 24-bit color, 8-bit gray, 1-bit binary?
*Answers:* color 1024·768·3 = **2,359,296 B (2.25 MB)**, gray **786,432 B (768 KB)**, binary 786,432/8 = **98,304 B (96 KB)**.

**N5. PSNR.** MSE between original and compressed = 100. PSNR?
*Answer:* 10·log10(255²/100) = 10·log10(650.25) = **28.13 dB**. Higher PSNR = closer to the original.

**N6. Point transforms.** (a) Negative of r = 60. (b) Log transform of r = 15 with `c = 255/ln(256)`. (c) Gamma 0.5 of r = 64.
*Answers:* (a) 255 - 60 = **195**. (b) c = 45.99, s = 45.99·ln(16) = 45.99 · 2.7726 = **127.5**. (c) 255·(64/255)^0.5 = 255 · 0.501 = **128**. Note how dark values got lifted a lot, that is the point of log and gamma < 1.

**N7. KNN by hand.** Training: (1,1) Red, (2,2) Red, (3,3) Red, (6,5) Blue, (7,7) Blue. Classify (4,4) with K = 3 using Euclidean distance.
*Answers:* distances: (3,3) 1.41, (6,5) 2.24, (2,2) 2.83, (1,1) 4.24, (7,7) 4.24. Three nearest: Red, Blue, Red. Majority = **Red**. (K = 1 also gives Red. Using Manhattan gives the same neighbours here.)

**N8. K-Means by hand.** Points (1,1), (2,1), (4,3), (5,4). K = 2, initial centroids (1,1) and (5,4). Run to convergence. Final centroids and total squared error?
*Answers:* Iteration 1 assignment: (1,1), (2,1) to C1; (4,3), (5,4) to C2. New centroids **C1 = (1.5, 1), C2 = (4.5, 3.5)**. Iteration 2: same assignment, so converged. Squared error = 0.25+0.25 + 0.5+0.5 = **1.5**.

**N9. Metrics.** TP = 40, FP = 10, FN = 20, TN = 30. Accuracy, precision, recall, F1?
*Answers:* accuracy 70/100 = **0.70**, precision 40/50 = **0.80**, recall 40/60 = **0.667**, F1 = 2·0.8·0.667 / (0.8+0.667) = **0.727**.

**N10. Arithmetic on uint8.** Pixel a = 200, b = 100. (a) NumPy `a + b`. (b) `cv2.add`. (c) NumPy `b - a`. (d) `cv2.subtract(b, a)`. (e) `addWeighted(a, 0.7, b, 0.3, 0)`.
*Answers:* (a) **44** (300 mod 256). (b) **255**. (c) **156** (-100 mod 256). (d) **0**. (e) 140 + 30 = **170**.

**N11. Geometric transforms.** Point (10, 20). Rotate 90 degrees counter-clockwise about the origin, then scale by 1.5, then translate by (100, 50). Final point?
*Answers:* rotation: x' = x cos - y sin = -20, y' = x sin + y cos = 10 gives (-20, 10). Scale: (-30, 15). Translate: **(70, 65)**.

**N12. Output size.** 5x5 image, 3x3 kernel. Output size with (a) no padding ("valid"), (b) zero padding of 1 ("same")? General formula?
*Answers:* (a) **3x3**, (b) **5x5**. Output = (N - k + 2p)/stride + 1.

**N13. Convolution border.** Apply the 3x3 average to the top-left corner pixel (value 90) of an image whose top-left 2x2 block is `[[90,60],[30,30]]` with zero padding.
*Answer:* sum of the 4 real pixels = 90+60+30+30 = 210, other 5 are zero. 210/9 = **23.3**. (This is why zero padding darkens image borders.)

**N14. KNN K choice.** 100 training points, 2 classes (60 and 40). What happens as K goes to 100? 
*Answer:* every prediction becomes the majority class (60), accuracy drops to the class prior. K = 1 overfits (noise-sensitive), K = N underfits.

---

## PART 6: VIVA (answer out loud in 2 to 3 sentences)

**Image basics**
1. What is an image to a computer? *A 2D array (gray) or 3D array (color) of intensity values, typically uint8 (0 to 255).*
2. Why is OpenCV BGR? *Historical camera convention. Matplotlib expects RGB so convert.*
3. Why 0.299, 0.587, 0.114? *Human eye is most sensitive to green, least to blue. Weights sum to 1.*
4. Grayscale vs binary? *Gray has 256 levels, binary only 2 (0 and 255) after thresholding.*
5. What is a pixel's 4-neighbour and 8-neighbour set? *N, S, E, W vs those plus 4 diagonals.*
6. Three distances? *Euclidean sqrt(dx²+dy²), city-block |dx|+|dy|, chessboard max(|dx|,|dy|).*
7. What is interpolation and which methods exist? *Estimating new pixel values when resizing/rotating: nearest (fast, blocky), bilinear (smooth), bicubic (smoother, slower).*

**Transforms**
8. Negative, log, gamma: what does each do? *Negative inverts. Log expands dark, compresses bright. Gamma < 1 brightens, > 1 darkens.*
9. Why `addWeighted` weights ~1? *Otherwise overall brightness changes. alpha·A + (1-alpha)·B.*
10. What does AND/OR/XOR do on binary images? *AND = intersection, OR = union, XOR = pixels in exactly one.*
11. What is a histogram? What does a narrow one mean? *Count of pixels at each intensity. Narrow means low contrast.*
12. How does histogram equalization work? *PDF, then CDF, then map each level to round((L-1)·CDF). Spreads intensities.*
13. Why equalize only V in color images? *Equalizing B, G, R separately changes color ratios, giving color distortion. V is brightness only.*
14. Lossy vs lossless? *JPEG/WEBP (lossy) discard information for smaller size. PNG is lossless: compression level changes size and speed, never pixels.*
15. Why does JPEG show blocks at low quality? *Works on 8x8 DCT blocks, quantization throws away high frequencies per block.*

**Filtering**
16. What is a kernel/convolution? *Small weight matrix slid over the image; each output is the weighted sum of the neighbourhood.*
17. Why odd kernel sizes? *Need a center pixel.*
18. What happens at borders? *Pad (zero, replicate, reflect). Zero padding darkens borders.*
19. Average vs Gaussian vs median? *Average: uniform blur, loses edges. Gaussian: center-weighted, smoother and more natural. Median: non-linear, best for salt-and-pepper, preserves edges.*
20. Why is median not a convolution? *It sorts and picks a value, it is not a weighted sum.*
21. What does the Laplacian detect and why does it amplify noise? *Second derivative, responds to rapid intensity change (edges); noise is rapid change too. Blur first.*
22. Sobel vs Laplacian? *Sobel: first derivative, directional (gx, gy), gives magnitude and direction. Laplacian: second derivative, no direction, sensitive to noise.*
23. How does sharpening work? *Subtract Laplacian (or add the difference between image and its blur) to boost edges.*
24. Why is blur not reversible? *Information is averaged away.*
25. Increasing kernel size does what? *More blur, more computation, edges and details lost.*

**K-Means**
26. Steps of K-Means? *Init K centroids, assign each point to the nearest, recompute centroids as means, repeat until stable.*
27. Is it supervised? *No, it is unsupervised, labels are not used.*
28. How to choose K? *Elbow method (inertia vs K), silhouette score, or domain knowledge.*
29. Limitations? *Must choose K, sensitive to initial centroids (use `n_init`, k-means++), assumes round similar-size clusters, sensitive to outliers and scale.*
30. What is inertia? *Sum of squared distances of points to their cluster centroids. Lower is tighter.*
31. What do cluster centers mean in image K-Means? *The dominant colors.*
32. Why does K-Means reduce image size when saved as PNG? *Fewer unique colors means much more repetition, so lossless compression works better.*
33. Does K-Means always converge? *To a local minimum, yes, but not necessarily the global best.*

**KNN**
34. How does KNN work? *Find K nearest training points, majority vote. Lazy learner, no training.*
35. Supervised? *Yes.*
36. Effect of K? *Small K: noisy, overfits. Large K: smooth, underfits. Use odd K for binary problems to avoid ties.*
37. Why scale features? *Distance is dominated by large-range features.*
38. KNN vs K-Means? *KNN supervised classification with labeled data. K-Means unsupervised clustering. The K means different things (neighbours vs clusters).*
39. Complexity/weakness? *Prediction is slow (distance to all points), needs memory, suffers in high dimensions.*
40. Metrics? *Confusion matrix, accuracy, precision (TP/(TP+FP)), recall (TP/(TP+FN)), F1 (harmonic mean).*
41. Why train/test split? *To measure performance on unseen data and detect overfitting.*

---

## PART 7: EXAM-HALL TACTICS

1. **Read twice, underline the blocks.** "Add noise, filter, then cluster" = noise + filter + K-Means. Do each block, print or show output after each.
2. **Get something running early.** `print(img.shape)` after every step. A half-working correct pipeline beats a perfect one that never ran.
3. **If asked "without library":** write loops or NumPy slicing, then compare with the library result using `np.abs(a.astype(int)-b.astype(int)).max()`. Printing that number impresses him because it shows you understand.
4. **No image in the exam?** Make one: `img = np.random.randint(0,256,(200,200,3),dtype=np.uint8)`. Data: `load_iris`, `load_digits`, `make_blobs` need no files.
5. **Forgot a function name?** `help(cv2.xyz)`, or type `cv2.` and press Tab.
6. **Explain while coding.** He wants the core idea. Say what each step does and why, in one sentence.
7. **If he asks "what happens inside this function?"** Use Part 2 and Part 3 ideas: kernel sliding, histogram to CDF to LUT, nearest K points voting, centroid update loop.
8. **Cast before arithmetic, clip after.** `x.astype(np.float32)` ... `np.clip(x,0,255).astype(np.uint8)`.
9. **Save outputs** with `cv2.imwrite` and show them; he may ask for the saved file.

---

## LAST CHECK BEFORE YOU SLEEP (can you do these with a blank file?)
- [ ] Logo overlay with both borders (B1) and one variation (B1b)
- [ ] `conv2d` from scratch, then average, Sobel, Laplacian built on it (D2, D5, D8)
- [ ] Median filter from scratch (D4) and the "median vs average on salt-and-pepper" explanation
- [ ] Histogram equalization from scratch (C3)
- [ ] Gamma, log, negative (B8 to B10)
- [ ] K-Means on CSV with plots for K = 2, 3, 4 (F1) and K-Means on pixels (F3, F4)
- [ ] K-Means from scratch (F2)
- [ ] KNN from scratch on Iris with all metrics (G1) and KNN on pixels (G5)
- [ ] Numericals N1, N2, N7, N8, N9, N10 by hand