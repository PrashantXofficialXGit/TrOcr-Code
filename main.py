import streamlit as st
from preprocessing import extract_text_from_image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import os
from PIL import Image

# ✅ Set your local model path
MODEL_PATH = r"D:/trocr_handwriting_model/trocr_handwriting_model"

# ✅ Check if the model exists before loading
if not os.path.exists(os.path.join(MODEL_PATH, "pytorch_model.bin")):
    raise FileNotFoundError(f"❌ Model file not found in {MODEL_PATH}")

# ✅ Load the processor and model from the local directory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = TrOCRProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH, local_files_only=True).to(device)

st.title("Handwriting OCR")

# ✅ Upload image
uploaded_file = st.file_uploader("Upload a handwritten note", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # ✅ Open the image using PIL
    image = Image.open(uploaded_file).convert("RGB")

    # ✅ Extract text using the preprocessing function
    extracted_text = extract_text_from_image(image, processor, model, device)

    # ✅ Display the extracted text
    st.subheader("Extracted Text:")
    st.write(extracted_text)
