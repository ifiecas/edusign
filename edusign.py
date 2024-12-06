import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import requests
from gtts import gTTS
import tempfile
import av

# Page Configuration
st.set_page_config(page_title="EduSign@VU: Sign Language for All", layout="wide", page_icon="🖐️")

# Initialize session states
if 'transcription_text' not in st.session_state:
    st.session_state.transcription_text = ""
if 'usage_count' not in st.session_state:
    st.session_state.usage_count = 0
if 'user_level' not in st.session_state:
    st.session_state.user_level = "Beginner"
if 'current_gesture' not in st.session_state:
    st.session_state.current_gesture = None
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None
if 'current_confidence' not in st.session_state:
    st.session_state.current_confidence = None

# Sidebar Navigation
st.sidebar.markdown(
    """
    <div style="text-align: center; margin-bottom: 20px;">
        <img src="https://i.postimg.cc/sgGLzYJV/Learn-Sign-Language.png" 
             style="width: 80%; height: auto;" alt="Sidebar Image">
    </div>
    """,
    unsafe_allow_html=True
)
page = st.sidebar.radio("Choose your learning path:", ["Home", "Sign Language Tutor", "Sign Language to Text", "Connect to a Mentor"])

# Load Machine Learning Model
@st.cache_resource
def load_model():
    model_url = "https://edusignstorage.blob.core.windows.net/model/sign_language_model_ver5.h5"
    try:
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_file:
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_file_path = temp_file.name
        model = tf.keras.models.load_model(temp_file_path)
        return model, True
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, False

gesture_model, model_loaded = load_model()

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
            "Maintain a gentle motion",
            "Ensure palm starts near lips",
            "Focus on smooth movement"
        ],
        "mistakes": [
            "Starting too far from lips",
            "Moving too sharply",
            "Incorrect finger position"
        ]
    },
    "Yes": {
        "steps": [
            "Make a fist",
            "Position in front of body",
            "Move up and down like nodding"
        ],
        "tips": [
            "Keep fist relaxed",
            "Use consistent rhythm",
            "Make motion visible"
        ],
        "mistakes": [
            "Exaggerated motion",
            "Hand too low",
            "Moving entire arm"
        ]
    },
    "No": {
        "steps": [
            "Form 'V' with fingers",
            "Palm facing outward",
            "Move side-to-side gently"
        ],
        "tips": [
            "Use controlled motions",
            "Keep fingers straight",
            "Position near face"
        ],
        "mistakes": [
            "Moving too fast",
            "Fingers too spread",
            "Hand too far from face"
        ]
    }
}

class GestureTutorTransformer(VideoTransformerBase):
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                
                h, w, _ = img.shape
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
                    
                    st.session_state.current_prediction = prediction
                    st.session_state.current_confidence = confidence
                    
                    cv2.putText(img, f"{prediction} ({confidence:.2f})", 
                              (x_min, y_min - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

class TranscriptionTransformer(VideoTransformerBase):
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                
                h, w, _ = img.shape
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
                    
                    if prediction in ["Hello", "Thank You"] and confidence > 0.3:
                        st.session_state.transcription_text += f"{prediction} "
                    
                    cv2.putText(img, f"{prediction} ({confidence:.2f})", 
                              (x_min, y_min - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def evaluate_user_level():
    if st.session_state.usage_count < 10:
        st.session_state.user_level = "Beginner"
    elif st.session_state.usage_count < 30:
        st.session_state.user_level = "Intermediate"
    else:
        st.session_state.user_level = "Expert"

if page == "Home":
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 40px;">
            <img src="https://i.postimg.cc/wjSrs4tM/Blue-Gradient-Header-Banner-1.png" 
                 style="width: 100%; max-width: 1000px; height: auto;" alt="EduSign Header">
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
       """
       <div style="text-align: center; background-color: #f8f9fa; padding: 2.5rem 0; border-radius: 10px;">
           <h2 style="color: #0f2f76; font-size: 1.8rem; line-height: 1.5; margin: 0 auto; max-width: 800px; padding: 0 20px;">
               EduSign AI is an innovative educational platform developed in partnership with 
               <span style="color: #0f2f76; font-weight: 700;">Victoria University</span> and powered by 
               <span style="color: #0f2f76; font-weight: 700;">Microsoft Azure AI</span>. 
           </h2>
           <p style="color: #2a4494; font-size: 1.2rem; margin: 1rem auto; max-width: 700px; padding: 0 20px;">
               Our mission is to make sign language learning accessible, interactive, and effective through cutting-edge technology.
           </p>
       </div>
       """,
       unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
       st.markdown(
           """
           <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); text-align: center; height: 100%;">
               <span style="font-size: 2.5rem;">🎯</span>
               <h4 style="color: #0f2f76; margin: 1rem 0; font-size: 1.4rem;">Smart Learning</h4>
               <p style="color: #666; line-height: 1.5;">Advanced AI recognition provides real-time feedback on your signing technique</p>
           </div>
           """,
           unsafe_allow_html=True
       )
    
    with col2:
       st.markdown(
           """
           <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); text-align: center; height: 100%;">
               <span style="font-size: 2.5rem;">📱</span>
               <h4 style="color: #0f2f76; margin: 1rem 0; font-size: 1.4rem;">Instant Translation</h4>
               <p style="color: #666; line-height: 1.5;">Microsoft Azure-powered recognition for accurate sign-to-text conversion</p>
           </div>
           """,
           unsafe_allow_html=True
       )
    
    with col3:
       st.markdown(
           """
           <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); text-align: center; height: 100%;">
               <span style="font-size: 2.5rem;">🤝</span>
               <h4 style="color: #0f2f76; margin: 1rem 0; font-size: 1.4rem;">Expert Guidance</h4>
               <p style="color: #666; line-height: 1.5;">Learn from certified mentors and native signers in our community</p>
           </div>
           """,
           unsafe_allow_html=True
       )

    st.markdown(
       """
       <div style="text-align: center; margin-top: 2rem;">
           <p style="color: #666; font-size: 1.1rem; line-height: 1.6; margin: 0 auto; max-width: 800px;">
               Our platform combines Victoria University's expertise in sign language education with 
               Microsoft's advanced AI technology to create a comprehensive learning experience. 
           </p>
       </div>
       <footer style="text-align: center; margin-top: 4rem; padding: 2rem; background-color: #f8f9fa; border-radius: 10px;">
           <div style="max-width: 800px; margin: 0 auto;">
               <p style="color: #0f2f76; font-size: 1.1rem; font-weight: 500; margin-bottom: 0.5rem;">
                   Developed by Ivy Fiecas-Borjal
               </p>
               <p style="color: #2a4494; font-size: 1rem; margin-bottom: 1rem;">
                   For the Victoria University - Accessibility AI Hackathon 2024
               </p>
               <a href="https://ifiecas.com/" target="_blank" style="
                   display: inline-block;
                   color: white;
                   background-color: #0f2f76;
