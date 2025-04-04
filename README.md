# 📝 Handwritten Text Extraction using EasyOCR & TrOCR

This project combines the power of **EasyOCR** for detecting handwritten text regions and **TrOCR (Transformer-based OCR)** for accurately extracting text from those regions. It's especially useful for reading doctor's prescriptions, handwritten notes, or any form of freehand text.

---

## 🚀 Demo

https://user-images.githubusercontent.com/your_video_or_gif_demo_here.gif

> 📌 EasyOCR detects the handwritten regions and draws bounding boxes.  
> ✨ TrOCR extracts accurate text from each selected region.

---

## 🧠 Models Used

- 🔍 **[EasyOCR](https://github.com/JaidedAI/EasyOCR)**: Used to detect and localize handwritten regions in the image.
- 📖 **[TrOCR](https://huggingface.co/microsoft/trocr-base-handwritten)**: A transformer-based OCR model for extracting clean and accurate text from the cropped regions.

---

---

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/handwriting-ocr.git
cd handwriting-ocr

Requirements include:
easyocr

torch

torchvision

transformers

pillow

matplotlib

opencv-python

Options:
Argument	Description	Default
--image	Path to the input image	Required
--output_dir	Folder to save outputs	outputs/
--show	Show result in a popup window	True
--save_text	Save extracted text to a file	True

⚙️ How It Works
🧠 Step 1: EasyOCR Region Detection

Image is passed through EasyOCR.

Bounding boxes for detected text regions are drawn.

Only handwritten-like areas are filtered for TrOCR.

✂️ Step 2: Cropping Regions

Each bounding box is cropped from the image.

🤖 Step 3: TrOCR Text Extraction

Each cropped region is passed through the TrOCR model.

Output text is extracted and stored.

💾 Step 4: Output Generation

Final image with boxes and text labels is saved.

Extracted text is saved in .txt format.

