import cv2
from PIL import Image
import os
import numpy as np

def process_image(image_path):
    if not os.path.exists(image_path):
        print("Error: File not found!")
        return

    # Using OpenCV to read the image
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        print("Error: Could not load image using OpenCV.")
        return
    
    # Print original size and shape
    height, width, channels = img_cv.shape
    print(f"Original Image - Width: {width}, Height: {height}, Channels: {channels}")

    # Convert to grayscale
    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    print(f"Grayscale Image - Shape: {gray_img.shape}")

    # Convert to binary (thresholding)
    _, binary_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY)
    print("Binary Image Conversion Done.")

    # Scale the image (reduce size by 50%)
    scale_percent = 50  # Scaling factor
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)
    resized_img = cv2.resize(img_cv, (new_width, new_height), interpolation=cv2.INTER_AREA)
    print(f"Scaled Image - New Width: {new_width}, New Height: {new_height}")

    # Remove noise using Gaussian Blur
    denoised_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    print("Noise Reduction Done using Gaussian Blur.")

    # Display the images
    cv2.imshow("Grayscale Image", gray_img)
    cv2.imshow("Binary Image", binary_img)
    cv2.imshow("Scaled Image", resized_img)
    cv2.imshow("Denoised Image", denoised_img)
    
    # Wait for key press and close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_path = os.path.join("data", "car.jpg")  
process_image(image_path)
