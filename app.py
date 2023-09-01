import streamlit as st
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
from gtts import gTTS
import cv2
import numpy as np

# Load the model and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define generation parameters
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# Function to generate captions and play audio
def generate_and_play_caption(image):
    # Process the image and generate the caption
    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values.to(device)
    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    caption = preds[0].strip()

    # Display the generated caption
    st.write("📷 Generated Caption:", caption)

    # Convert caption to speech and save as audio file
    speech = gTTS(text=caption, lang='en')
    audio_file = "speech.mp3"
    speech.save(audio_file)

    # Play the audio using st.audio
    audio_data = open(audio_file, "rb").read()
    st.audio(audio_data, format="audio/mp3")

# Function to capture a single image from IVcam Pro
def capture_single_ivcam_image():
    # Set the camera index for IVcam Pro (adjust this index based on your system)
    cap = cv2.VideoCapture(1)  # Change the index (1, 2, etc.) if needed

    if not cap.isOpened():
        st.error("Error: IVcam Pro not found or cannot be opened.")
        return None

    st.write("📸 Capturing...")

    ret, frame = cap.read()
    if not ret:
        st.error("Error: Failed to capture image.")
        return None

    # Convert the OpenCV frame to a PIL image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Display the image
    st.image(pil_image, caption="Captured Image 📸", use_column_width=True)

    cap.release()
    return pil_image

# Streamlit app
st.title("VisionVoice 📸🔊")

uploaded_image = st.file_uploader(" Upload an image", type=["jpg", "jpeg", "png"])

# IVcam Pro button
if st.button("Use Camera "):
    ivcam_image = capture_single_ivcam_image()
    if ivcam_image:
        # Generate caption and play audio for the captured image
        generate_and_play_caption(ivcam_image)


if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image 🖼️", use_column_width=True)

    # Read the uploaded image
    image = Image.open(uploaded_image)
    if image.mode != "RGB":
        image = image.convert(mode="RGB")

    if st.button("Generate Caption"):
        generate_and_play_caption(image)

    if st.button("Play Audio Caption"):
        generate_and_play_caption(image)  # Reuse the function for audio caption



