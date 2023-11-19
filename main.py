import os
import time
import requests
import streamlit as st
from dotenv import load_dotenv
from camera_input_live import camera_input_live


def speak(text):
    os.system(f'say "{text}"')


def query(frame):
    with open(frame, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()


load_dotenv(os.path.join(os.path.dirname(os.path.curdir), ".env"))

# Set up the Hugging Face token here
API_URL = (
    "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
)
API_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
headers = {"Authorization": "Bearer {}".format(API_TOKEN)}


last_capture_time = time.time()
filename = ""

image = camera_input_live(
    key="eyeblip",
    start_label="Start",
    stop_label="Pause",
)

st.image(image=image)

current_time = time.time()
time_diff = current_time - last_capture_time

if image is not None:
    if time_diff >= 5:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"frame_{timestamp}.jpg"

        image_binary = image.getvalue()

        with open(filename, "wb") as f:
            f.write(image_binary)

        last_capture_time = current_time
        saved_frame = filename

        response = query(saved_frame)
        caption = response[0]["generated_text"]

        speak(caption)
        st.text(caption)

        # TODO: Enable this later after experiment complete
        # os.remove(saved_frame)
