## 💡 Running in Headless Environments (WSL, Docker, SSH)

Since this project is developed using **WSL (Windows Subsystem for Linux)** without a graphical user interface (GUI), OpenCV's image display function `cv2.imshow()` is disabled by default. 

Instead, the script saves outputs directly to the disk using **`cv2.imwrite()`**.

### 🖥️ Running on a Local Machine with a GUI?
If you are running this code on a standard desktop environment and want to view the images interactively using `cv2.imshow()`, remember to append the following clean-up functions to the end of your script to prevent the window from freezing:

```python
import cv2

# ... your OpenCV image processing code ...

cv2.imshow("Output Preview", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
