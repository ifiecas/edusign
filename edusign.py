import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from gtts import gTTS
import tempfile

# Page Configuration
st.set_page_config(page_title="EduSign - AI Powered Sign Language Tutor", layout="wide", page_icon="🖐️")

# Initialize session states
if 'webcam_running' not in st.session_state:
    st.session_state.webcam_running = False
if 'transcription_running' not in st.session_state:
    st.session_state.transcription_running = False
if 'transcription_text' not in st.session_state:
    st.session_state.transcription_text = ""

# Sidebar Navigation
st.sidebar.title("EduSign")
st.sidebar.markdown("### AI-Powered Sign Language Tutor")
st.sidebar.markdown("Empowering Communication Through Sign Language Learning")
page = st.sidebar.radio("Choose a feature", ["Sign Language Tutor", "Sign Language to Text"])

# Load Machine Learning Model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("/Users/raphael/signlanguage_tutor/sign_language_model_ver5.h5")
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
            "Position your hand near your lips",
            "Keep your fingers together and palm facing inward",
            "Extend your thumb slightly",
            "Move your hand outward and slightly downward"
        ],
        "tips": [
            "Maintain a gentle motion when moving your hand outward",
            "Ensure your palm starts near your lips and is visible to the viewer",
            "Focus on the smoothness of the gesture"
        ],
        "mistakes": [
            "Starting too far from your lips",
            "Moving your hand too sharply or abruptly",
            "Bending your fingers during the motion"
        ]
    },
    "Yes": {
        "steps": [
            "Make a fist with your dominant hand",
            "Position the fist in front of your body",
            "Move your fist up and down slightly, as if nodding"
        ],
        "tips": [
            "Keep your fist relaxed and avoid tensing your hand",
            "Maintain a consistent rhythm during the motion",
            "Ensure the motion is visible and clear to the viewer"
        ],
        "mistakes": [
            "Making the motion too large or exaggerated",
            "Keeping your hand too low to be seen",
            "Moving your entire arm instead of just the fist"
        ]
    },
    "No": {
        "steps": [
            "Extend your index and middle fingers, forming a 'V'",
            "Position your hand near your face, palm facing outward",
            "Move your hand side-to-side gently, as if shaking your head"
        ],
        "tips": [
            "Focus on small, controlled motions",
            "Ensure your fingers are straight and close together",
            "Keep your hand at a comfortable level near your face"
        ],
        "mistakes": [
            "Moving your hand too fast or erratically",
            "Spacing your fingers too far apart",
            "Positioning your hand too far away from your face"
        ]
    }
}

def detect_gesture(frame):
    """Detect gestures and return the gesture and confidence."""
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

def transcription_feed(frame_placeholder, transcription_placeholder):
    """Handle real-time transcription from the camera."""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Cannot access webcam. Please check your camera connection.")
        return

    try:
        while st.session_state.transcription_running:
            ret, frame = cap.read()
            if not ret:
                break

            frame, gesture, confidence = detect_gesture(frame)
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width="always")
            
            if gesture and confidence > 0.8:
                st.session_state.transcription_text += f"{gesture} "
                transcription_placeholder.markdown(
                    f'<div class="transcription-box">{st.session_state.transcription_text.strip()}</div>', 
                    unsafe_allow_html=True
                )
    finally:
        cap.release()

def process_frame(frame, selected_gesture):
    """Process the webcam frame to detect and classify gestures with feedback."""
    frame, prediction, confidence = detect_gesture(frame)
    feedback = None

    if prediction is None:
        feedback = "No hand detected. Make sure your hand is visible to the camera."
    elif prediction != selected_gesture:
        feedback = f"Try again! Remember, for '{selected_gesture}':\n" + "\n".join(
            [f"- {step}" for step in learning_guides[selected_gesture]["steps"]]
        )
    else:
        feedback = f"Great job! You've successfully signed '{selected_gesture}'."

    return frame, prediction, confidence, feedback

def start_webcam_feed(frame_placeholder, feedback_placeholder, selected_gesture):
    """Start the webcam feed for practicing gestures."""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Cannot access webcam. Please check your camera connection.")
        return

    try:
        while st.session_state.webcam_running:
            ret, frame = cap.read()
            if not ret:
                break

            frame, gesture, confidence, feedback = process_frame(frame, selected_gesture)
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if feedback_placeholder:
                with feedback_placeholder.container():
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Detected Gesture", gesture if gesture else "None")
                    with col2:
                        st.metric("Confidence", f"{confidence*100:.1f}%" if confidence else "N/A")
                    
                    st.markdown(f"### Feedback:\n{feedback}")
    finally:
        cap.release()

# EduSign - AI Powered Sign Language Tutor Page
if page == "Sign Language Tutor":
    st.title("🖐️ EduSign - AI Powered Sign Language Tutor")
    if not model_loaded:
        st.error("Model not loaded. Please check the model file and restart.")
    else:
        selected_gesture = st.selectbox("Select a word to learn:", list(gesture_classes.values()))
        
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
            guide = learning_guides.get(selected_gesture, {})
            
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
                start_webcam_feed(frame_placeholder, feedback_placeholder, selected_gesture)
            
            if st.button("Toggle Webcam"):
                st.session_state.webcam_running = not st.session_state.webcam_running
            st.markdown(f"Status: {'🟢 Active' if st.session_state.webcam_running else '🔴 Inactive'}")

# Sign Language to Text Page
elif page == "Sign Language to Text":
    st.title("🖐️ Sign Language to Text")

    if not model_loaded:
        st.error("Model not loaded. Please check the model file and restart.")
    else:
        st.markdown("""
        <style>
            .transcription-box {
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                font-size: 1.5rem;
                color: #333;
                text-align: center;
                margin-top: 20px;
            }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("### Real-Time Transcription")

        # Layout for webcam and transcription
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Webcam Feed")
            frame_placeholder = st.empty()

        with col2:
            st.markdown("### Transcribed Text")
            transcription_placeholder = st.markdown(
                '<div class="transcription-box">Waiting for transcription...</div>', 
                unsafe_allow_html=True
            )

        # Toggle transcription feed
        if st.session_state.transcription_running:
            transcription_feed(frame_placeholder, transcription_placeholder)

        if st.button("Toggle Transcription"):
            st.session_state.transcription_running = not st.session_state.transcription_running

        # Options for download and listen
        st.markdown("### Options")
        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("Download Transcription"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
                    tmp_file.write(st.session_state.transcription_text.encode())
                    st.download_button("Download", tmp_file.name, "transcription.txt")
        
        with col2:
            if st.button("Listen to Transcription"):
                tts = gTTS(st.session_state.transcription_text.strip())
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                    tts.save(tmp_file.name)
                    st.audio(tmp_file.name)
