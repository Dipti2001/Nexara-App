import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Initialize MediaPipe Holistic model and drawing utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Load the pre-trained model
model = load_model('action.h5')

# Define the actions you want to recognize
actions = ['hello', 'imagine', 'cup']  # Example actions

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results
    
def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])
    
def prob_viz(res, actions, input_frame, colors):
    # First flip the frame
    flipped_frame = cv2.flip(input_frame, 1)
    
    # Now, work with the flipped frame for further processing
    output_frame = flipped_frame.copy()
    
    for num, prob in enumerate(res):
        # Since the frame is flipped, the text needs to be added in the mirrored position.
        text_location = (output_frame.shape[1] - int(prob*100), 85+num*40) # Adjust text location

        # Draw the rectangle for the background
        cv2.rectangle(output_frame, (output_frame.shape[1] - int(prob*100), 60+num*40), (output_frame.shape[1], 90+num*40), colors[num], -1)
        
        # Add the text on top of the rectangle
        cv2.putText(output_frame, actions[num], text_location, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    # After adding the text and graphics, return the flipped frame with the readable text
    return output_frame

# Helper function to process and predict on frames
def process_frame(frame, sequence, holistic):
    # Process the frame with MediaPipe and custom function
    image, results = mediapipe_detection(frame, holistic)
    draw_styled_landmarks(image, results)

    # Extract keypoints for the current frame
    keypoints = extract_keypoints(results)
    sequence.append(keypoints)
    sequence = sequence[-30:]  # Keep the last 30

    # Check if we have a sequence of 30 to predict
    if len(sequence) == 30:
        sequence_array = np.expand_dims(np.array(sequence), axis=0)  # Reshape for the model
        res = model.predict(sequence_array)[0]  # Predict
        image = prob_viz(res, actions, image, [(245,117,16), (117,245,16), (16,117,245)])

    return image, sequence