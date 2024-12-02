import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from datetime import datetime
import pyttsx3

# Page Configuration
st.set_page_config(page_title="EduSign", layout="wide", page_icon="🖐️")

# Initialize session states
if 'webcam_running' not in st.session_state:
    st.session_state.webcam_running = False

# Sidebar Navigation
st.sidebar.title("EduSign")
st.sidebar.markdown("### Empowering Communication Through Sign Language Learning")
option = st.sidebar.radio(
    "Choose a Learning Path",
    ["Home", "Sign Language Tutor", "VUTranscribe - Convert Text to Speech"]
)

# Load Machine Learning Model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("/Users/raphael/signlanguage_tutor/sign_language_model_ver4.h5")
        return model, True
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, False

gesture_model, model_loaded = load_model()

# MediaPipe Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

# Gesture Classes and Learning Guides
gesture_classes = {0: "Hello", 1: "Thank You", 2: "Yes", 3: "No"}

learning_guides = {
    "Hello": {
        "steps": [
            "Position your hand near your forehead",
            "Keep your palm facing outward",
            "Extend your fingers naturally",
            "Move your hand away in a slight arc"
        ],
        "tips": [
            "Keep movements fluid and relaxed",
            "Maintain a comfortable distance from your forehead",
            "Practice the motion slowly at first"
        ],
        "mistakes": [
            "Keeping hand too far from forehead",
            "Making movements too rigid",
            "Tensing fingers unnecessarily"
        ]
    },
    # Add other gesture learning guides here...
}

def process_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    prediction = None
    confidence = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            h, w, _ = frame.shape
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            
            x_min, x_max = map(int, [min(x_coords), max(x_coords)])
            y_min, y_max = map(int, [min(y_coords), max(y_coords)])
            
            padding = 20
            x_min, x_max = max(0, x_min - padding), min(w, x_max + padding)
            y_min, y_max = max(0, y_min - padding), min(h, y_max + padding)
            
            if x_min < x_max and y_min < y_max:
                hand_img = frame_rgb[y_min:y_max, x_min:x_max]
                hand_img = cv2.resize(hand_img, (224, 224))
                hand_img = hand_img / 255.0
                
                pred = gesture_model.predict(np.expand_dims(hand_img, axis=0), verbose=0)
                prediction = gesture_classes.get(np.argmax(pred))
                confidence = float(np.max(pred))

    return frame, prediction, confidence

def start_webcam_feed(frame_placeholder, feedback_placeholder=None):
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Cannot access webcam. Please check your camera connection.")
        return

    try:
        while st.session_state.webcam_running:
            ret, frame = cap.read()
            if not ret:
                break

            frame, gesture, confidence = process_frame(frame)
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if feedback_placeholder:
                with feedback_placeholder.container():
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Gesture", gesture if gesture else "None")
                    with col2:
                        st.metric("Confidence", f"{confidence*100:.1f}%" if confidence else "N/A")
    finally:
        cap.release()

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Home Page
if option == "Home":
    st.title("Welcome to EduSign!")
    st.subheader("Learn and Connect with Sign Language")
    st.markdown("""
    EduSign offers two amazing paths to explore:
    - 🖐️ **Sign Language Tutor**: Practice gestures and receive real-time feedback
    - 🖋️ **VUTranscribe**: Convert text to speech
    """)

# Sign Language Tutor Page
elif option == "Sign Language Tutor":
    st.title("🖐️ Sign Language Tutor")
    
    if not model_loaded:
        st.error("Model not loaded. Please check the model file and restart.")
    else:
        word_to_learn = st.selectbox("Select a word to learn:", list(gesture_classes.values()))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Tutorial Video")
            st.markdown("""
                <div style="width: 100%; position: relative; padding-top: 56.25%;">
                    <iframe 
                        src="https://www.youtube.com/embed/iRsWS96g1B8"
                        style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"
                        frameborder="0" 
                        allowfullscreen>
                    </iframe>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("### Learning Guide")
            guide = learning_guides.get(word_to_learn, {})
            
            st.markdown("#### Steps:")
            for step in guide.get("steps", []):
                st.markdown(f"• {step}")
            
            st.markdown("#### Pro Tips:")
            for tip in guide.get("tips", []):
                st.markdown(f"• {tip}")
            
            st.markdown("#### Common Mistakes:")
            for mistake in guide.get("mistakes", []):
                st.markdown(f"• {mistake}")
        
        with col2:
            st.markdown("### Practice Area")
            frame_placeholder = st.empty()
            feedback_placeholder = st.empty()
            
            if st.session_state.webcam_running:
                start_webcam_feed(frame_placeholder, feedback_placeholder)
            
            if st.button("Toggle Webcam"):
                st.session_state.webcam_running = not st.session_state.webcam_running
            st.markdown(f"Status: {'🟢 Active' if st.session_state.webcam_running else '🔴 Inactive'}")

# VUTranscribe Page
elif option == "VUTranscribe - Convert Text to Speech":
    st.title("🖋️ VUTranscribe")
    st.markdown("Convert your text into speech with ease.")

    input_text = st.text_area("Enter text to convert to speech:")
    if st.button("Convert to Speech"):
        if input_text:
            text_to_speech(input_text)
            st.success("Text has been converted to speech!")
        else:
            st.warning("Please enter some text first.")
