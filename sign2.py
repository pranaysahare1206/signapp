import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from googletrans import Translator
import time

# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize Translator
translator = Translator()

# Streamlit layout
st.set_page_config(layout="wide")
st.title("Enhanced Sign Language Detection App")

# Sidebar
st.sidebar.title("Settings")

# Language selection dropdown
language = st.sidebar.selectbox("Select Language", ["Marathi", "Hindi", "English"])

# Gesture type selection
gesture_type = st.sidebar.radio("Select Gesture Type", ["Alphabets", "Numbers", "Words/Sentences"])

# Two columns in Streamlit
col1, col2 = st.columns([3, 1])

with col1:
    st.header("Real-Time Video")
    FRAME_WINDOW = st.image([])

with col2:
    st.header("Translation")
    output_text = st.empty()

# Dictionary for sign language gestures and their translations
signs = {
    # Alphabets
    "A": "A", "B": "B", "C": "C", "D": "D", "E": "E", "F": "F", "G": "G", "H": "H", "I": "I", "J": "J",
    "K": "K", "L": "L", "M": "M", "N": "N", "O": "O", "P": "P", "Q": "Q", "R": "R", "S": "S", "T": "T",
    "U": "U", "V": "V", "W": "W", "X": "X", "Y": "Y", "Z": "Z",
    # Numbers
    "0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five",
    "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine", "10": "Ten",
    # Basic words/sentences
    "Hello": "Hello",
    "Thank You": "Thank You",
    "Goodbye": "Goodbye",
    "Yes": "Yes",
    "No": "No",
    "Help": "I need help",
    "Sorry": "I'm sorry",
    "Please": "Please",
    "Excuse Me": "Excuse Me",
    "Love": "Love",
    "Family": "Family",
    "Friend": "Friend",
    "Food": "Food",
    "Water": "Water",
    "More": "More",
    "Stop": "Stop",
    "Again": "Again",
    "How Are You": "How Are You",
    "I Am Fine": "I Am Fine",
    "What Is Your Name": "What Is Your Name",
    "My Name Is": "My Name Is",
    "Where Is The Bathroom": "Where Is The Bathroom"
}

def detect_hand_gestures(landmarks):
    # Helper function to check if a finger is extended
    def is_finger_extended(finger_tip, finger_base, palm_base):
        return landmarks[finger_tip].y < landmarks[finger_base].y < landmarks[palm_base].y

    # Helper function to calculate angle between three points
    def calculate_angle(a, b, c):
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return angle if angle <= 180 else 360 - angle

    if gesture_type == "Alphabets":
        # Detect alphabets
        if is_finger_extended(8, 6, 0) and not any(is_finger_extended(i, i - 2, 0) for i in [12, 16, 20]):
            return "A"
        elif all(is_finger_extended(i, i - 2, 0) for i in [8, 12, 16, 20]) and landmarks[4].y > landmarks[3].y:
            return "B"
        elif calculate_angle(landmarks[0], landmarks[9], landmarks[3]) < 60:
            return "C"
        elif is_finger_extended(8, 6, 0) and all(not is_finger_extended(i, i - 2, 0) for i in [12, 16, 20]):
            return "D"
        elif all(not is_finger_extended(i, i - 2, 0) for i in [8, 12, 16, 20]):
            return "E"
        elif is_finger_extended(12, 10, 0) and all(not is_finger_extended(i, i - 2, 0) for i in [8, 16, 20]):
            return "F"
        elif is_finger_extended(8, 6, 0) and is_finger_extended(16, 14, 0) and not is_finger_extended(12, 10, 0):
            return "G"
        elif is_finger_extended(12, 10, 0) and not is_finger_extended(8, 6, 0):
            return "H"
        elif is_finger_extended(8, 6, 0) and all(not is_finger_extended(i, i - 2, 0) for i in [12, 16, 20]) and \
                landmarks[4].x < landmarks[3].x:
            return "I"
        elif is_finger_extended(8, 6, 0) and not is_finger_extended(12, 10, 0) and landmarks[4].x < landmarks[3].x:
            return "J"
        elif is_finger_extended(10, 8, 0) and not any(is_finger_extended(i, i - 2, 0) for i in [16, 20]):
            return "K"
        elif is_finger_extended(8, 6, 0) and not any(is_finger_extended(i, i - 2, 0) for i in [12, 16, 20]):
            return "L"
        elif all(not is_finger_extended(i, i - 2, 0) for i in [8, 12, 16, 20]) and landmarks[4].x < landmarks[3].x:
            return "M"
        elif all(not is_finger_extended(i, i - 2, 0) for i in [8, 12, 16]) and is_finger_extended(20, 18, 0):
            return "N"
        elif calculate_angle(landmarks[0], landmarks[5], landmarks[8]) > 90:
            return "O"
        elif is_finger_extended(8, 6, 0) and is_finger_extended(12, 10, 0) and all(
                not is_finger_extended(i, i - 2, 0) for i in [16, 20]):
            return "P"
        elif is_finger_extended(8, 6, 0) and is_finger_extended(16, 14, 0) and landmarks[4].x < landmarks[3].x:
            return "Q"
        elif is_finger_extended(8, 6, 0) and is_finger_extended(12, 10, 0) and not any(
                is_finger_extended(i, i - 2, 0) for i in [16, 20]):
            return "R"
        elif all(not is_finger_extended(i, i - 2, 0) for i in [8, 12, 16, 20]) and landmarks[4].y > landmarks[3].y:
            return "S"
        elif is_finger_extended(8, 6, 0) and not is_finger_extended(12, 10, 0):
            return "T"
        elif is_finger_extended(8, 6, 0) and is_finger_extended(12, 10, 0) and not any(
                is_finger_extended(i, i - 2, 0) for i in [16, 20]):
            return "U"
        elif is_finger_extended(8, 6, 0) and all(not is_finger_extended(i, i - 2, 0) for i in [16, 20]):
            return "V"
        elif is_finger_extended(8, 6, 0) and is_finger_extended(12, 10, 0) and is_finger_extended(16, 14, 0):
            return "W"
        elif not is_finger_extended(8, 6, 0) and all(not is_finger_extended(i, i - 2, 0) for i in [12, 16, 20]):
            return "X"
        elif is_finger_extended(12, 10, 0) and not any(is_finger_extended(i, i - 2, 0) for i in [8, 16, 20]):
            return "Y"
        elif is_finger_extended(8, 6, 0) and not any(is_finger_extended(i, i - 2, 0) for i in [12, 16, 20]):
            return "Z"


    elif gesture_type == "Numbers":

        # Detect numbers

        if all(not is_finger_extended(i, i - 2, 0) for i in [8, 12, 16, 20]) and landmarks[4].y < landmarks[3].y:

            return "0"

        elif is_finger_extended(8, 6, 0) and not any(is_finger_extended(i, i - 2, 0) for i in [12, 16, 20]):

            return "1"

        elif all(is_finger_extended(i, i - 2, 0) for i in [8, 12]) and not any(
                is_finger_extended(i, i - 2, 0) for i in [16, 20]):

            return "2"

        elif all(is_finger_extended(i, i - 2, 0) for i in [8, 12, 16]) and not is_finger_extended(20, 18, 0):

            return "3"

        elif all(is_finger_extended(i, i - 2, 0) for i in [8, 12, 16, 20]):

            return "4"

        elif all(is_finger_extended(i, i - 2, 0) for i in [8, 12, 16, 20]) and landmarks[4].y > landmarks[3].y:

            return "5"

        elif is_finger_extended(8, 6, 0) and is_finger_extended(12, 10, 0) and not any(
                is_finger_extended(i, i - 2, 0) for i in [16, 20]):

            return "6"

        elif is_finger_extended(8, 6, 0) and is_finger_extended(16, 14, 0) and not any(
                is_finger_extended(i, i - 2, 0) for i in [12, 20]):

            return "7"

        elif is_finger_extended(8, 6, 0) and is_finger_extended(12, 10, 0) and is_finger_extended(16, 14,
                                                                                                  0) and not is_finger_extended(
                20, 18, 0):

            return "8"

        elif is_finger_extended(8, 6, 0) and is_finger_extended(12, 10, 0) and is_finger_extended(16, 14,
                                                                                                  0) and is_finger_extended(
                20, 18, 0):

            return "9"

        elif all(is_finger_extended(i, i - 2, 0) for i in [8, 12, 16, 20]) and landmarks[4].y < landmarks[3].y:

            return "10"



    elif gesture_type == "Words/Sentences":

        # Detect basic words/sentences

        if landmarks[8].y < landmarks[6].y and landmarks[4].y > landmarks[3].y:

            return "Hello"

        elif landmarks[4].y < landmarks[3].y and landmarks[8].y < landmarks[6].y:

            return "Thank You"

        elif landmarks[12].y < landmarks[11].y:

            return "Goodbye"

        elif landmarks[8].y < landmarks[7].y and landmarks[4].y > landmarks[3].y:

            return "Yes"

        elif landmarks[4].y < landmarks[3].y and landmarks[12].y > landmarks[11].y:

            return "No"

        elif is_finger_extended(8, 6, 0) and is_finger_extended(4, 3, 0) and not any(
                is_finger_extended(i, i - 2, 0) for i in [12, 16, 20]):

            return "I need help"

        elif all(not is_finger_extended(i, i - 2, 0) for i in [8, 12, 16, 20]) and landmarks[4].y < landmarks[3].y:

            return "I'm sorry"

        elif landmarks[4].y > landmarks[3].y and landmarks[8].y > landmarks[6].y:

            return "Please"

        elif landmarks[4].y < landmarks[3].y and landmarks[8].y > landmarks[6].y and landmarks[12].y < landmarks[11].y:

            return "Excuse Me"

        elif landmarks[4].y > landmarks[3].y and landmarks[8].y > landmarks[6].y and landmarks[16].y < landmarks[14].y:

            return "Love"

        elif landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[11].y and landmarks[16].y > landmarks[
            14].y:

            return "Family"

        elif landmarks[4].y < landmarks[3].y and all(is_finger_extended(i, i - 2, 0) for i in [8, 12, 16]):

            return "Friend"

        elif all(not is_finger_extended(i, i - 2, 0) for i in [12, 16]) and landmarks[8].y < landmarks[6].y:

            return "Food"

        elif landmarks[4].y < landmarks[3].y and landmarks[8].y < landmarks[6].y and landmarks[16].y > landmarks[14].y:

            return "Water"

        elif all(is_finger_extended(i, i - 2, 0) for i in [8, 12]) and landmarks[16].y > landmarks[14].y:

            return "More"

        elif landmarks[8].y > landmarks[6].y and landmarks[4].y < landmarks[3].y:

            return "Stop"

        elif landmarks[12].y < landmarks[11].y and landmarks[4].y < landmarks[3].y:

            return "Again"

        elif all(is_finger_extended(i, i - 2, 0) for i in [8, 12, 16]) and landmarks[4].y > landmarks[3].y:

            return "How Are You"

        elif all(not is_finger_extended(i, i - 2, 0) for i in [8, 12]) and landmarks[4].y < landmarks[3].y:

            return "I Am Fine"

        elif landmarks[4].y > landmarks[3].y and landmarks[8].y > landmarks[6].y and landmarks[12].y < landmarks[11].y:

            return "What Is Your Name"

        elif landmarks[8].y < landmarks[6].y and landmarks[4].y > landmarks[3].y and landmarks[12].y > landmarks[11].y:

            return "My Name Is"

        elif landmarks[12].y > landmarks[11].y and landmarks[16].y > landmarks[14].y and landmarks[8].y < landmarks[
            6].y:

            return "Where Is The Bathroom"

    return None



def process_frame(image, hands):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    gesture = None
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = detect_hand_gestures(hand_landmarks.landmark)
            if gesture:
                break
    
    return image, gesture

def main():
    # Initialize MediaPipe Hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    previous_gesture = None
    gesture_start_time = None
    gesture_timeout = 1  # Time in seconds to wait before finalizing a gesture

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video. Please check your webcam.")
            break

        # Process frame and detect gesture
        frame, gesture = process_frame(frame, hands)

        # Display the video feed
        FRAME_WINDOW.image(frame, channels="BGR")

        # Handle gesture detection and translation
        if gesture:
            current_time = time.time()
            if gesture == previous_gesture:
                if gesture_start_time and (current_time - gesture_start_time) >= gesture_timeout:
                    translation = translator.translate(signs.get(gesture, "Unknown"), dest=language.lower()).text
                    output_text.markdown(
                        f"<h2>Gesture Detected:</h2><h1>{gesture}</h1><h2>Translation in {language}:</h2><h1>{translation}</h1>",
                        unsafe_allow_html=True
                    )
                    previous_gesture = None
                    gesture_start_time = None
            else:
                previous_gesture = gesture
                gesture_start_time = current_time

        # Check if the user wants to stop the application
        if st.button('Stop'):
            break

    # Release resources
    cap.release()
    hands.close()

if __name__ == "__main__":
    main()
