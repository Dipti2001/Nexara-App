import cv2
import streamlit as st
from model_utils import process_frame, mp_holistic
import numpy as np

# Set up the Streamlit interface
st.set_page_config(layout="wide")
st.title("ASL Detection ML Model (Interactive Application)")



# Use the beta_columns feature to create a 2-column layout
col1, col2 = st.columns(2)

# Set up the left column with the user profile and camera feed
with col1:
      # Replace with the path to your user profile image
    
    st.write("Video Detection time: 3O frames")
    # Start and stop buttons
    start_button_pressed = st.button("Start")
    stop_button_pressed = st.button("Stop")

    # Placeholder for the video frames
    frame_placeholder = st.empty()

# Set up the right column with the device status
with col2:
    
    st.write("Sign To Try: ")
    st.image('hello.png', width=250)  # Replace with the path to your device image
    st.write("Hello")
    st.image('imagine.png', width=250)  # Replace with the path to your device image
    st.write("Imagine")
    st.image('cup.png', width=250)  # Replace with the path to your device image
    st.write("Cup")

# Initialize the video capture object
cap = None

# Main loop for the video feed
if start_button_pressed:
    cap = cv2.VideoCapture(1)
    sequence = []  # Initialize an empty list for storing sequences
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened() and not stop_button_pressed:
            ret, frame = cap.read()
            if not ret:
                st.write("The video capture has ended.")
                break
            
            # Process the frame and make predictions
            image, sequence = process_frame(frame, sequence, holistic)

            # Mirror the frame
            # image = cv2.flip(image, 1)

            # Convert the colors from BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Display the frame in the left column Streamlit app
            frame_placeholder.image(image, channels="RGB")

# Release the video capture object and destroy all OpenCV windows when the stop button is pressed
if stop_button_pressed and cap is not None:
    cap.release()
    st.write("Video capture stopped.")
