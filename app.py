import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoTransformerBase
import cv2
from model_utils import process_frame, mp_holistic
import av

# Set up the Streamlit interface
st.set_page_config(layout="wide")
st.title("ASL Detection ML Model (Interactive Application)")

# Use the beta_columns feature to create a 2-column layout
col1, col2 = st.columns(2)

# Set up the right column with the device status
with col2:
    st.write("Sign To Try: ")
    st.image('hello.png', width=250)  # Example path to your device image
    st.write("Hello")
    st.image('imagine.png', width=250)  # Example path to your device image
    st.write("Imagine")
    st.image('cup.png', width=250)  # Example path to your device image
    st.write("Cup")

# Define the RTC configuration (optional, for STUN/TURN servers)
rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Define a VideoTransformer class to process video frames
class ASLTransformer(VideoTransformerBase):
    def __init__(self):
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Process the frame with your ML model or custom logic
        processed_image, _ = process_frame(img, [], self.holistic)
        
        # Convert the colors from BGR to RGB
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        
        # Convert the processed image back to an AV frame
        new_frame = av.VideoFrame.from_ndarray(processed_image, format="rgb24")
        return new_frame

# Set up the left column with the user profile and camera feed
with col1:
    st.write("Video Detection time: 30 frames")
    
    # Start the WebRTC streamer
    webrtc_ctx = webrtc_streamer(key="example", 
                                 video_transformer_factory=ASLTransformer,
                                 rtc_configuration=rtc_configuration)

# Additional logic can be added here, for example, to handle stop conditions or display results
