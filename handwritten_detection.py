import cv2
import numpy as np
from PIL import Image

def detect_handwritten_areas(image):
    """
    Detect handwritten text regions in an image and return cropped handwritten text areas.

    Parameters:
        - image (str or PIL.Image or np.ndarray): The input image (file path, PIL Image, or NumPy array).

    Returns:
        - list of np.ndarray: A list of cropped handwritten text regions.
    """

    # ✅ Ensure input is a NumPy array (handle file paths & PIL images)
    if isinstance(image, str):  # If image is a file path
        image = cv2.imread(image)
    elif isinstance(image, Image.Image):  # If image is a PIL image
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR for OpenCV

    # ✅ Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ✅ Apply adaptive thresholding for better handwritten text detection
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # ✅ Apply morphological operations to connect broken characters
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # ✅ Find contours of the text regions
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    handwritten_regions = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > 10 and w > 10:  # Ignore small noise
            roi = gray[y:y+h, x:x+w]  # Extract text region
            handwritten_regions.append(roi)  # Store cropped handwritten text areas

    return handwritten_regions
