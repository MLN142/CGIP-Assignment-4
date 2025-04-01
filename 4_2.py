import cv2
import os
import numpy as np

def process_image(image_path):
    if not os.path.exists(image_path):
        print("Error: File not found!")
        return

    # Load the image
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        print("Error: Could not load image using OpenCV.")
        return
    
    # Convert to grayscale (needed for some operations)
    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # 1. Inverse Transformation (Image Negative)
    inverse_img = cv2.bitwise_not(img_cv)
    print("Inverse Transformation Done.")

    # 2. Contrast Stretching (Min-Max Normalization)
    img_min = np.min(gray_img)
    img_max = np.max(gray_img)
    contrast_stretched_img = ((gray_img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    print("Contrast Stretching Done.")

    # 3. Histogram Equalization (Enhancing Contrast)
    hist_eq_img = cv2.equalizeHist(gray_img)
    print("Histogram Equalization Done.")

    # 4. Edge Detection (Canny Edge Detector)
    edges = cv2.Canny(gray_img, 100, 200)
    print("Edge Detection Done.")

    # Display the processed images
    cv2.imshow("Inverse Image", inverse_img)
    cv2.imshow("Contrast Stretched Image", contrast_stretched_img)
    cv2.imshow("Histogram Equalized Image", hist_eq_img)
    cv2.imshow("Edge Detection", edges)

    # Wait for key press and close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_path = os.path.join("data", "car.jpg") 
process_image(image_path)
