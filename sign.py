import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from googletrans import Translator
from gtts import gTTS
import io
import pygame
import time
import requests
from PIL import Image

# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Initialize Translator
translator = Translator()

# Initialize pygame for sound playback
pygame.mixer.init()

# Streamlit layout
st.set_page_config(layout="wide")
st.title("संवर्धित सांकेतिक भाषा शोध अॅप")  # Enhanced Sign Language Detection App in Marathi

# Sidebar
st.sidebar.title("सेटिंग्ज")  # Settings in Marathi

# Language selection dropdown
language = st.sidebar.selectbox("भाषा निवडा", ["मराठी", "हिंदी", "इंग्रजी"])  # Select Language in Marathi

# Gesture type selection
gesture_type = st.sidebar.radio("हावभाव प्रकार निवडा",
                                ["अक्षरे", "संख्या", "शब्द/वाक्ये"])  # Select Gesture Type in Marathi


# Function to convert text to speech using gTTS and play it using pygame
def text_to_speech(text, language):
    lang_code = {'मराठी': 'mr', 'हिंदी': 'hi', 'इंग्रजी': 'en'}
    tts = gTTS(text=text, lang=lang_code[language])
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    pygame.mixer.music.load(mp3_fp, 'mp3')
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    pygame.mixer.music.stop()


# Two columns in Streamlit
col1, col2 = st.columns([3, 1])

with col1:
    st.header("रीअल-टाइम व्हिडिओ")  # Real-Time Video in Marathi
    FRAME_WINDOW = st.image([])

with col2:
    st.header("अनुवाद आणि भाषण")  # Translation & Speech in Marathi
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
def process_frame(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process hands
    hand_results = hands.process(image_rgb)

    gesture = None

    # Process hand landmarks and detect gestures
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = detect_hand_gestures(hand_landmarks.landmark)
            if gesture:
                break  # Stop after detecting the first gesture

    return image, gesture


# Function to search for sign language images online
def search_sign_language_image(query):
    api_key = "AIzaSyC67Ql12-qf0gu7vRlXQXfFVcvNsO0UOOw"  # Replace with your actual API key
    search_engine_id = "050e33f9000b24ed6"  # Replace with your actual search engine ID
    url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={search_engine_id}&q={query}&searchType=image"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'items' in data and len(data['items']) > 0:
            return data['items'][0]['link']
    return None


# Start video capture
cap = cv2.VideoCapture(0)

# Set a larger frame size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 520)

# Real-time Video Processing
previous_gesture = None
gesture_timeout = 1  # Time in seconds to wait before finalizing a gesture
gesture_start_time = None
unidentified_count = 0
last_identified_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        st.write("व्हिडिओ कॅप्चर त्रुटी")  # Video Capture Error in Marathi
        break

    # Process frame and detect gesture
    frame, gesture = process_frame(frame)

    # Display the video feed (larger size)
    FRAME_WINDOW.image(frame, channels="BGR", use_column_width=True)

    # Handle gesture detection and translation
    current_time = time.time()
    if gesture:
        if gesture == previous_gesture:
            if gesture_start_time and (current_time - gesture_start_time) >= gesture_timeout:
                with col2:
                    translation = translator.translate(signs.get(gesture, "अज्ञात"), dest=language.lower()).text
                    output_text.markdown(
                        f"<h2>ओळखलेला हावभाव:</h2><h1>{gesture}</h1><h2>{language} मध्ये अनुवाद:</h2><h1>{translation}</h1>",
                        unsafe_allow_html=True)
                    text_to_speech(translation, language)
                previous_gesture = None
                gesture_start_time = None
                unidentified_count = 0
                last_identified_time = current_time
        else:
            previous_gesture = gesture
            gesture_start_time = current_time
    else:
        if current_time - last_identified_time > 5:  # 5 seconds of no identification
            unidentified_count += 1
            if unidentified_count >= 2:
                with col2:
                    instruction = translator.translate("कृपया काही सेकंद स्थिर राहा किंवा हावभाव पुन्हा करा",
                                                       dest=language.lower()).text
                    output_text.markdown(f"<h2>{instruction}</h2>", unsafe_allow_html=True)
                    text_to_speech(instruction, language)

                    if unidentified_count >= 3:
                        # Capture the current frame
                        _, img_encoded = cv2.imencode('.jpg', frame)
                        img_bytes = img_encoded.tobytes()

                        # Search for sign language image online
                        search_result = search_sign_language_image("sign language " + gesture_type)
                        if search_result:
                            image_url = search_result
                            response = requests.get(image_url)
                            img = Image.open(io.BytesIO(response.content))
                            st.image(img, caption="संभाव्य सांकेतिक भाषा हावभाव", use_column_width=True)

                        unidentified_count = 0

            last_identified_time = current_time

    # Introduce a short delay to improve performance and avoid flickering
    time.sleep(0.1)

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()