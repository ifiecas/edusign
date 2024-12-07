import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import requests
from gtts import gTTS
import tempfile
import av
import time

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
if 'feedback_text' not in st.session_state:
    st.session_state.feedback_text = ""
if 'last_transcribed_gesture' not in st.session_state:
    st.session_state.last_transcribed_gesture = None
if 'real_time_gesture' not in st.session_state:
    st.session_state.real_time_gesture = ""
if 'real_time_confidence' not in st.session_state:
    st.session_state.real_time_confidence = None

# Constants
CONFIDENCE_THRESHOLD = 0.29  # 29% confidence threshold
TRANSCRIPTION_THRESHOLD = 0.29  # 70% confidence for transcription

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

# Load model and setup
gesture_model, model_loaded = load_model()

# Gesture Classes
gesture_classes = {0: "Hello", 1: "Thank You", 2: "Yes", 3: "No"}

# Learning guides for each gesture
learning_guides = {
    "Hello": {
        "steps": ["Position hand near forehead", "Palm facing outward", "Extend fingers naturally", "Move hand away in arc"],
        "tips": ["Keep movements fluid", "Maintain comfortable distance", "Practice slowly at first"],
        "mistakes": ["Hand too far from forehead", "Movements too rigid", "Tense fingers"]
    },
    "Thank You": {
        "steps": ["Hand near lips", "Fingers together, palm inward", "Extend thumb", "Move outward and down"],
        "tips": ["Gentle motion", "Start near lips", "Smooth movement"],
        "mistakes": ["Too far from lips", "Moving too sharp", "Wrong finger position"]
    },
    "Yes": {
        "steps": ["Make a fist", "Position in front", "Move up and down"],
        "tips": ["Keep relaxed", "Consistent rhythm", "Clear motion"],
        "mistakes": ["Too exaggerated", "Hand too low", "Moving whole arm"]
    },
    "No": {
        "steps": ["Form 'V' with fingers", "Palm outward", "Side-to-side motion"],
        "tips": ["Control movement", "Straight fingers", "Near face"],
        "mistakes": ["Too fast", "Spread fingers", "Hand too far"]
    }
}

def get_feedback_style(confidence):
    if confidence is None:
        return "🤚 Show your hand to the camera"
    elif confidence <= CONFIDENCE_THRESHOLD:
        return "error"
    elif confidence < 0.29:
        return "warning"
    else:
        return "success"

def generate_gesture_feedback(prediction, confidence, target_gesture):
    if prediction is None:
        return "No hand detected. Please show your hand to the camera.", "error"
    
    if confidence <= CONFIDENCE_THRESHOLD:
        mistakes = learning_guides[target_gesture]["mistakes"]
        tips = learning_guides[target_gesture]["tips"]
        feedback = f"Low confidence ({confidence:.1%}). Please check:\n\n"
        feedback += "Common mistakes to avoid:\n"
        for mistake in mistakes:
            feedback += f"• {mistake}\n"
        feedback += "\nTry these tips:\n"
        for tip in tips:
            feedback += f"• {tip}\n"
        return feedback, "error"
    
    elif prediction != target_gesture:
        feedback = f"Currently showing: '{prediction}'\nTrying to learn: '{target_gesture}'\n\n"
        feedback += "Tips for the correct gesture:\n"
        for step in learning_guides[target_gesture]["steps"]:
            feedback += f"• {step}\n"
        return feedback, "warning"
    
    elif confidence < 0.7:
        tips = learning_guides[target_gesture]["tips"]
        feedback = f"Good attempt! ({confidence:.1%})\n\nTips to improve:\n"
        for tip in tips:
            feedback += f"• {tip}\n"
        return feedback, "warning"
    
    else:
        return f"Excellent! Perfect '{target_gesture}' sign! ({confidence:.1%})", "success"

class GestureTutorProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.29,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.last_prediction_time = 0
        self.prediction_cooldown = 0.2

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        current_time = time.time()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

                h, w, _ = img.shape
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                
                x_min, x_max = map(int, [min(x_coords), max(x_coords)])
                y_min, y_max = map(int, [min(y_coords), max(y_coords)])
                
                padding = 20
                x_min = max(0, x_min - padding)
                x_max = min(w, x_max + padding)
                y_min = max(0, y_min - padding)
                y_max = min(h, y_max + padding)
                
                if current_time - self.last_prediction_time >= self.prediction_cooldown:
                    if x_min < x_max and y_min < y_max:
                        hand_img = rgb_frame[y_min:y_max, x_min:x_max]
                        hand_img = cv2.resize(hand_img, (224, 224))
                        hand_img = hand_img / 255.0
                        
                        try:
                            pred = gesture_model.predict(np.expand_dims(hand_img, axis=0), verbose=0)
                            prediction = gesture_classes.get(np.argmax(pred))
                            confidence = float(np.max(pred))
                            
                            # Update session states
                            st.session_state.current_prediction = prediction
                            st.session_state.current_confidence = confidence
                            
                            # Get feedback
                            feedback, feedback_type = generate_gesture_feedback(
                                prediction, confidence, st.session_state.current_gesture
                            )
                            st.session_state.feedback_text = feedback
                            
                            # Display on frame
                            color = (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (0, 0, 255)
                            label = f"Detected: {prediction} ({confidence:.1%})"
                            cv2.putText(img, label, (x_min, y_min - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                            
                            self.last_prediction_time = current_time
                        except Exception as e:
                            print(f"Prediction error: {e}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

class TranscriptionProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.29,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.last_prediction_time = 0
        self.prediction_cooldown = 0.5

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        current_time = time.time()
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                h, w, _ = img.shape
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                
                x_min, x_max = map(int, [min(x_coords), max(x_coords)])
                y_min, y_max = map(int, [min(y_coords), max(y_coords)])
                
                padding = 20
                x_min = max(0, x_min - padding)
                x_max = min(w, x_max + padding)
                y_min = max(0, y_min - padding)
                y_max = min(h, y_max + padding)
                
                if current_time - self.last_prediction_time >= self.prediction_cooldown:
                    if x_min < x_max and y_min < y_max:
                        hand_img = rgb_frame[y_min:y_max, x_min:x_max]
                        hand_img = cv2.resize(hand_img, (224, 224))
                        hand_img = hand_img / 255.0
                        
                        try:
                            pred = gesture_model.predict(np.expand_dims(hand_img, axis=0), verbose=0)
                            prediction = gesture_classes.get(np.argmax(pred))
                            confidence = float(np.max(pred))
                            
                            # Update real-time detection
                            st.session_state.real_time_gesture = prediction
                            st.session_state.real_time_confidence = confidence
                            
                            # Add to transcription if confidence is high enough
                            if confidence > TRANSCRIPTION_THRESHOLD:
                                if (prediction != st.session_state.last_transcribed_gesture or 
                                    current_time - self.last_prediction_time > 2.0):
                                    st.session_state.transcription_text += f"{prediction} "
                                    st.session_state.last_transcribed_gesture = prediction
                                    self.last_prediction_time = current_time
                            
                            # Display on frame
                            color = (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (0, 0, 255)
                            label = f"Detected: {prediction} ({confidence:.1%})"
                            cv2.putText(img, label, (x_min, y_min - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        except Exception as e:
                            print(f"Prediction error: {e}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# Page Implementations
if page == "Home":
    # [Home page implementation remains exactly the same]
    pass

elif page == "Sign Language Tutor":
    st.title("🖐️ EduSign - Your Sign Language Tutor")

    if not model_loaded:
        st.error("Model failed to load. Please check the URL and restart the application.")
    else:
        selected_gesture = st.selectbox("Select a word to learn:", list(gesture_classes.values()))
        st.session_state.current_gesture = selected_gesture

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Learning Guide")
            
            st.markdown("""
                <div style="width: 100%; padding-bottom: 75%; position: relative;">
                    <iframe 
                        src="https://www.youtube.com/embed/Sdw7a-gQzcU"
                        style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"
                        frameborder="0" 
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                        allowfullscreen>
                    </iframe>
                </div>
                """, unsafe_allow_html=True)
            
            guide = learning_guides.get(selected_gesture, {})
            st.markdown("#### Steps to perform this gesture:")
            for step in guide.get("steps", []):
                st.markdown(f"- {step}")

        with col2:
            st.markdown("### Practice Area")
            webrtc_ctx = webrtc_streamer(
                key="gesture-tutor",
                video_processor_factory=GestureTutorProcessor,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )

            # Real-time detection display
            if webrtc_ctx.state.playing:
                st.markdown("### Current Detection")
                detection_col1, detection_col2 = st.columns(2)
                with detection_col1:
                    st.metric(
                        "Detected Gesture",
                        st.session_state.current_prediction if st.session_state.current_prediction else "None"
                    )
                with detection_col2:
                    confidence = st.session_state.current_confidence
                    if confidence:
                        color = "red" if confidence <= CONFIDENCE_THRESHOLD else "green"
                        st.markdown(f'<p style="color: {color}; font-size: 1.2em;">Confidence: {confidence:.1%}</p>', 
                                  unsafe_allow_html=True)

                # Display feedback with appropriate styling
                st.markdown("### EduSign AI's Feedback")
                if st.session_state.feedback_text:
                    if confidence and confidence <= CONFIDENCE_THRESHOLD:
                        st.error(st.session_state.feedback_text)
                    elif confidence and confidence < 0.7:
                        st.warning(st.session_state.feedback_text)
                    else:
                        st.success(st.session_state.feedback_text)

elif page == "Sign Language to Text":
    st.title("🖐️ Gesture Translator | Converting Sign Language to Text")
    
    st.markdown("""
    <style>
        .transcription-box {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            font-size: 1.5rem;
            color: #333;
            margin-top: 20px;
        }
        .detection-box {
            background-color: #e3f2fd;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            font-size: 1.2rem;
        }
    </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Webcam Feed")
        webrtc_ctx = webrtc_streamer(
            key="transcription",
            video_processor_factory=TranscriptionProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        # Real-time detection display beside video
        if webrtc_ctx.state.playing:
            st.markdown("### Current Detection")
            detection_text = "No gesture detected"
            if st.session_state.real_time_gesture and st.session_state.real_time_confidence:
                detection_text = f"Detecting: {st.session_state.real_time_gesture} ({st.session_state.real_time_confidence:.1%})"
            st.markdown(f'<div class="detection-box">{detection_text}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("### Transcribed Text")
        st.markdown(
            f'<div class="transcription-box">{st.session_state.transcription_text or "Waiting for signs..."}</div>',
            unsafe_allow_html=True
        )

        button_col1, button_col2 = st.columns(2)
        with button_col1:
            if st.button("Clear Text"):
                st.session_state.transcription_text = ""
                st.experimental_rerun()
            
            if st.button("Download Text"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
                    tmp_file.write(st.session_state.transcription_text.encode())
                    st.download_button("Download", tmp_file.name, "transcription.txt")

        with button_col2:
            if st.button("Listen to Text"):
                if st.session_state.transcription_text.strip():
                    tts = gTTS(st.session_state.transcription_text.strip())
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                        tts.save(tmp_file.name)
                        with open(tmp_file.name, "rb") as audio_file:
                            st.audio(audio_file.read(), format="audio/mp3")
                else:
                    st.warning("No text to read.")

elif page == "Connect to a Mentor":
    st.title("🖐️ Connect to a Mentor")
    
    mentors = {
        "Beginner": {
            "Alex": "Specializes in foundational signs and building confidence.",
            "Sarah": "Expert in teaching basics and proper hand positioning.",
        },
        "Intermediate": {
            "Jordan": "Helps with fluency and conversational signing.",
            "Maya": "Focuses on vocabulary expansion and complex gestures.",
        },
        "Expert": {
            "Taylor": "Advanced signing techniques and professional interpretation.",
            "Raj": "Specialized in technical and medical sign language.",
        }
    }

    st.markdown(
        f"""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h3>Your Current Level: <span style="color: #0f2f76;">{st.session_state.user_level}</span></h3>
            <p>Usage Count: {st.session_state.usage_count} sessions</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    level_mentors = mentors.get(st.session_state.user_level, {})
    selected_mentor = st.selectbox("Choose your mentor:", list(level_mentors.keys()))
    preferred_time = st.select_slider("Select preferred time:", 
                                    options=["9:00 AM", "10:00 AM", "11:00 AM", "2:00 PM", "3:00 PM", "4:00 PM"])

    if st.button("Schedule Session"):
        st.session_state.usage_count += 1
        st.success(f"✅ Session scheduled with {selected_mentor} at {preferred_time}")
        st.balloons()

    # Display mentor details
    st.markdown(
        f"""
        <div style="background-color: white; padding: 20px; border-radius: 10px; margin-top: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h3 style="color: #0f2f76;">About Your Mentor</h3>
            <p style="font-size: 1.1rem;">{level_mentors.get(selected_mentor)}</p>
            <p style="margin-top: 15px; color: #666;">Available for {st.session_state.user_level} level students</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Progress tracking
    st.markdown(
        """
        <div style="margin-top: 30px;">
            <h3 style="color: #0f2f76;">Your Learning Journey</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    progress = min(st.session_state.usage_count / 30 * 100, 100)  # Calculate progress percentage
    st.progress(progress / 100)
    
    next_level = "Intermediate" if st.session_state.user_level == "Beginner" else "Expert"
    sessions_needed = 30 - st.session_state.usage_count if st.session_state.usage_count < 30 else 0
    
    if sessions_needed > 0:
        st.info(f"Complete {sessions_needed} more sessions to reach {next_level} level!")
    else:
        st.success("Congratulations! You've reached the highest level!")

    st.markdown(
        """
        <footer style="text-align: center; margin-top: 4rem; padding: 2rem; background-color: #f8f9fa; border-radius: 10px;">
            <div style="max-width: 800px; margin: 0 auto;">
                <p style="color: #0f2f76; font-size: 1.1rem; font-weight: 500; margin-bottom: 0.5rem;">
                    Developed by Ivy Fiecas-Borjal
                </p>
                <p style="color: #2a4494; font-size: 1rem; margin-bottom: 1rem;">
                    For the Victoria University - Accessibility AI Hackathon 2024
                </p>
                <a href="https://ifiecas.com/" target="_blank" style="display: inline-block; color: white; background-color: #0f2f76; padding: 0.5rem 1.5rem; border-radius: 25px; text-decoration: none;">View Portfolio</a>
            </div>
        </footer>
        """,
        unsafe_allow_html=True
    )
