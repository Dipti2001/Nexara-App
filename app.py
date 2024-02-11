import cv2
import streamlit as st
from pathlib import Path
from model_utils import process_frame, mp_holistic
import numpy as np

# Set up the Streamlit interface
st.set_page_config(layout="wide")
st.title("ASL Detection ML Model (Interactive Application)")

# Use the beta_columns feature to create a 2-column layout
col1, col2 = st.columns(2)

# Set up the left column
with col1:
    st.write("Video Detection time: 30 frames")
    start_button_pressed = st.button("Start")
    stop_button_pressed = st.button("Stop")
    frame_placeholder = st.empty()

# Set up the right column
with col2:
    st.write("Sign To Try: ")
    st.image('hello.png', width=250)
    st.write("Hello")
    st.image('imagine.png', width=250)
    st.write("Imagine")
    st.image('cup.png', width=250)
    st.write("Cup")

# Function to find a working camera
def find_working_camera():
    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        if cap is None or not cap.isOpened():
            cv2.destroyAllWindows()
            cap.release()
            index += 1
        else:
            return cap, index

# Initialize the video capture object
cap = None

# Main loop for the video feed
if start_button_pressed:
    cap, camera_index = find_working_camera()  # Dynamically find a working camera
    if cap is None:
        st.write("No working camera found.")
    else:
        st.write(f"Using camera {camera_index}.")
        sequence = []
        
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened() and not stop_button_pressed:
                ret, frame = cap.read()
                if not ret:
                    st.write("The video capture has ended.")
                    break
                
                # Process the frame and make predictions
                image, sequence = process_frame(frame, sequence, holistic)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(image, channels="RGB")
