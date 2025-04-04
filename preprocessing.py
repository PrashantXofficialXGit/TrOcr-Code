from PIL import Image
import torch
import numpy as np
from handwritten_detection import detect_handwritten_areas

from PIL import Image

def extract_text_from_image(image_path, processor, model, device):
    """
    Extract text from detected handwritten text regions in the image.
    """
    # Detect handwritten text areas
    handwritten_regions = detect_handwritten_areas(image_path)

    extracted_texts = []

    for region in handwritten_regions:
        # Convert cropped region to PIL image and ensure it's in RGB format
        roi_pil = Image.fromarray(region).convert("RGB")  # 🔥 Convert grayscale to RGB
        
        # Process image using the processor
        pixel_values = processor(roi_pil, return_tensors="pt").pixel_values.to(device)

        # Generate text prediction
        with torch.no_grad():
            output = model.generate(pixel_values)

        # Decode the predicted text
        text = processor.batch_decode(output, skip_special_tokens=True)[0]
        extracted_texts.append(text)

    return "\n".join(extracted_texts)
