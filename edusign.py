import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from datetime import datetime
import speech_recognition as sr
import threading
import queue
import time

# Page Configuration
st.set_page_config(page_title="EduSign", layout="wide", page_icon="🖐️")

# Initialize session states
if 'webcam_running' not in st.session_state:
    st.session_state.webcam_running = False
if 'transcription_history' not in st.session_state:
    st.session_state.transcription_history = []
if 'audio_queue' not in st.session_state:
    st.session_state.audio_queue = queue.Queue()

# Sidebar Navigation
st.sidebar.title("EduSign")
st.sidebar.markdown("### Empowering Communication Through Sign Language Learning")
option = st.sidebar.radio(
    "Choose a Learning Path",
    ["Home", "Sign Language Tutor", "Attend Online Classes"]
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
    "Thank You": {
        "steps": [
            "Start with your open hand near your chin",
            "Keep your palm facing yourself",
            "Move your hand forward and down",
            "End with a slight bow of your head"
        ],
        "tips": [
            "Keep the motion smooth and graceful",
            "Maintain eye contact when appropriate",
            "Don't rush the movement"
        ],
        "mistakes": [
            "Moving hand too quickly",
            "Forgetting to tilt head slightly",
            "Making motion too small"
        ]
    },
    "Yes": {
        "steps": [
            "Form a fist with your hand",
            "Hold your hand up near your head",
            "Nod your hand up and down",
            "Keep your wrist firm"
        ],
        "tips": [
            "Make the nodding motion clear",
            "Keep your arm relatively still",
            "Maintain a steady rhythm"
        ],
        "mistakes": [
            "Making motion too subtle",
            "Moving entire arm instead of wrist",
            "Nodding too fast or slow"
        ]
    },
    "No": {
        "steps": [
            "Extend your index and middle fingers",
            "Keep other fingers closed",
            "Move your hand side to side",
            "Keep movement controlled"
        ],
        "tips": [
            "Keep the motion horizontal",
            "Maintain consistent speed",
            "Don't overextend the movement"
        ],
        "mistakes": [
            "Using too many fingers",
            "Making diagonal movements",
            "Moving too vigorously"
        ]
    }
}

def process_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    prediction = None
    confidence = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Process hand region for prediction
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

def start_webcam_feed(frame_placeholder, feedback_placeholder=None, mode="tutor"):
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Cannot access webcam. Please check your camera connection.")
        return

    # Speech recognition setup for class mode
    if mode == "class":
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 4000
        
    try:
        while st.session_state.webcam_running:
            ret, frame = cap.read()
            if not ret:
                break

            if mode == "tutor":
                frame, gesture, confidence = process_frame(frame)
                frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                if feedback_placeholder:
                    with feedback_placeholder.container():
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Gesture", gesture if gesture else "None")
                        with col2:
                            st.metric("Confidence", f"{confidence*100:.1f}%" if confidence else "N/A")
            
            else:  # Class mode
                frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                try:
                    with sr.Microphone() as source:
                        audio = recognizer.listen(source, phrase_time_limit=1, timeout=0.1)
                        st.session_state.audio_queue.put(audio)
                except (sr.WaitTimeoutError, TimeoutError):
                    continue
                except Exception as e:
                    print(f"Audio capture error: {e}")
                    continue

    finally:
        cap.release()

def process_audio_queue():
    recognizer = sr.Recognizer()
    while st.session_state.webcam_running:
        try:
            if not st.session_state.audio_queue.empty():
                audio = st.session_state.audio_queue.get_nowait()
                try:
                    text = recognizer.recognize_google(audio)
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    st.session_state.transcription_history.append(f"{timestamp}: {text}")
                except sr.UnknownValueError:
                    pass
                except Exception as e:
                    print(f"Transcription error: {e}")
        except queue.Empty:
            time.sleep(0.1)

# Home Page
if option == "Home":
    st.title("Welcome to EduSign!")
    st.subheader("Learn and Connect with Sign Language")
    st.markdown("""
    EduSign offers two amazing paths to explore:
    - 🖐️ **Sign Language Tutor**: Practice gestures and receive real-time feedback
    - 🎓 **Attend Online Classes**: Get real-time speech-to-text transcription
    """)

# Sign Language Tutor Page
elif option == "Sign Language Tutor":
    st.title("🖐️ Sign Language Tutor")
    
    if not model_loaded:
        st.error("Model not loaded. Please check the model file and restart.")
    else:
        word_to_learn = st.selectbox("Select a word to learn:", list(gesture_classes.values()))
        
        # Container for better alignment
        with st.container():
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Tutorial Video")
                # Custom CSS for iframe container
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
                
                # Learning Guide
                st.markdown("### Learning Guide")
                guide = learning_guides[word_to_learn]
                
                st.markdown("#### Steps:")
                for step in guide["steps"]:
                    st.markdown(f"• {step}")
                
                st.markdown("#### Pro Tips:")
                for tip in guide["tips"]:
                    st.markdown(f"• {tip}")
                
                st.markdown("#### Common Mistakes:")
                for mistake in guide["mistakes"]:
                    st.markdown(f"• {mistake}")
            
            with col2:
                st.markdown("### Practice Area")
                # Webcam feed placeholder
                frame_placeholder = st.empty()
                feedback_placeholder = st.empty()
                
                if st.session_state.webcam_running:
                    start_webcam_feed(frame_placeholder, feedback_placeholder, mode="tutor")
                
                # Controls below webcam feed
                if st.button("Toggle Webcam"):
                    st.session_state.webcam_running = not st.session_state.webcam_running
                st.markdown(f"Status: {'🟢 Active' if st.session_state.webcam_running else '🔴 Inactive'}")

# Attend Online Classes Page
elif option == "Attend Online Classes":
    st.title("🎓 Attend Online Classes")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Video Feed")
        frame_placeholder = st.empty()
        
        if st.button("Toggle Webcam"):
            st.session_state.webcam_running = not st.session_state.webcam_running
            if st.session_state.webcam_running:
                # Start audio processing thread
                audio_thread = threading.Thread(
                    target=process_audio_queue,
                    daemon=True
                )
                audio_thread.start()
        
        st.markdown(f"Status: {'🟢 Active' if st.session_state.webcam_running else '🔴 Inactive'}")
        
        if st.session_state.webcam_running:
            start_webcam_feed(frame_placeholder, mode="class")
    
    with col2:
        st.markdown("### Live Transcription")
        
        # Control buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Clear"):
                st.session_state.transcription_history.clear()
        with col_btn2:
            if st.button("Download"):
                transcript = "\n".join(st.session_state.transcription_history)
                st.download_button(
                    "Save Transcript",
                    transcript,
                    file_name=f"transcript_{datetime.now():%Y%m%d_%H%M%S}.txt",
                    mime="text/plain"
                )
        
        # Live transcription display
        transcript_container = st.container()
        with transcript_container:
            for text in reversed(st.session_state.transcription_history[-10:]):
                st.text(text)