import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image in grayscale
image = cv2.imread('/content/drive/MyDrive/Application photo.jpg',cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not load image.")
    exit()

# Apply a binary threshold to the image
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Define a kernel for the morphological operations
kernel = np.ones((5, 5), np.uint8)

# Apply morphological operations
dilated = cv2.dilate(binary_image, kernel, iterations=1)
eroded = cv2.erode(binary_image, kernel, iterations=1)
opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

# Plot the results
plt.figure(figsize=(12, 10))

plt.subplot(2, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title('Dilated Image')
plt.imshow(dilated, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title('Eroded Image')
plt.imshow(eroded, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title('Opened Image')
plt.imshow(opened, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title('Closed Image')
plt.imshow(closed, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
