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


# Video Transformer Classes
class GestureTutorTransformer(VideoTransformerBase):
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def transform(self, frame):
        # Convert from WebRTC frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process hands
        results = self.hands.process(rgb_frame)
        
        # Draw hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                self.mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Get hand ROI
                h, w, _ = img.shape
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                
                x_min, x_max = map(int, [min(x_coords), max(x_coords)])
                y_min, y_max = map(int, [min(y_coords), max(y_coords)])
                
                # Add padding
                padding = 20
                x_min = max(0, x_min - padding)
                x_max = min(w, x_max + padding)
                y_min = max(0, y_min - padding)
                y_max = min(h, y_max + padding)
                
                if x_min < x_max and y_min < y_max:
                    # Extract and preprocess hand image
                    hand_img = rgb_frame[y_min:y_max, x_min:x_max]
                    hand_img = cv2.resize(hand_img, (224, 224))
                    hand_img = hand_img / 255.0
                    
                    try:
                        # Predict gesture
                        pred = gesture_model.predict(np.expand_dims(hand_img, axis=0), verbose=0)
                        prediction = gesture_classes.get(np.argmax(pred))
                        confidence = float(np.max(pred))
                        
                        # Update session state
                        st.session_state.current_prediction = prediction
                        st.session_state.current_confidence = confidence
                        
                        # Draw prediction on frame
                        label = f"{prediction}: {confidence:.2f}"
                        cv2.putText(img, label, (x_min, y_min - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    except Exception as e:
                        print(f"Prediction error: {e}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

class TranscriptionTransformer(VideoTransformerBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.last_prediction_time = 0
        self.prediction_cooldown = 1.0  # Seconds between predictions

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
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
                
                current_time = time.time()
                if current_time - self.last_prediction_time >= self.prediction_cooldown:
                    if x_min < x_max and y_min < y_max:
                        hand_img = rgb_frame[y_min:y_max, x_min:x_max]
                        hand_img = cv2.resize(hand_img, (224, 224))
                        hand_img = hand_img / 255.0
                        
                        try:
                            pred = gesture_model.predict(np.expand_dims(hand_img, axis=0), verbose=0)
                            prediction = gesture_classes.get(np.argmax(pred))
                            confidence = float(np.max(pred))
                            
                            if prediction in ["Hello", "Thank You"] and confidence > 0.3:
                                st.session_state.transcription_text += f"{prediction} "
                                self.last_prediction_time = current_time
                            
                            label = f"{prediction}: {confidence:.2f}"
                            cv2.putText(img, label, (x_min, y_min - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        except Exception as e:
                            print(f"Prediction error: {e}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Page Implementations
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

    # Main content
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            """
            <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); text-align: center;">
                <span style="font-size: 2.5rem;">🎯</span>
                <h4 style="color: #0f2f76; margin: 1rem 0;">Smart Learning</h4>
                <p style="color: #666;">AI-powered real-time feedback</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            """
            <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); text-align: center;">
                <span style="font-size: 2.5rem;">📱</span>
                <h4 style="color: #0f2f76; margin: 1rem 0;">Instant Translation</h4>
                <p style="color: #666;">Sign-to-text conversion</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            """
            <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); text-align: center;">
                <span style="font-size: 2.5rem;">🤝</span>
                <h4 style="color: #0f2f76; margin: 1rem 0;">Expert Guidance</h4>
                <p style="color: #666;">Learn from certified mentors</p>
            </div>
            """,
            unsafe_allow_html=True
        )

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
            guide = learning_guides.get(selected_gesture, {})
            
            st.markdown("#### Steps:")
            for step in guide.get("steps", []):
                st.markdown(f"- {step}")
            
            st.markdown("#### Pro Tips:")
            for tip in guide.get("tips", []):
                st.markdown(f"- {tip}")
            
            st.markdown("#### Common Mistakes:")
            for mistake in guide.get("mistakes", []):
                st.markdown(f"- {mistake}")

        with col2:
            st.markdown("### Practice Area")
            webrtc_ctx = webrtc_streamer(
                key="gesture-tutor",
                video_transformer_factory=GestureTutorTransformer,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )

            if webrtc_ctx.state.playing:
                feedback_container = st.empty()
                with feedback_container.container():
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.metric("Detected Gesture", 
                                st.session_state.current_prediction if st.session_state.current_prediction else "None")
                    with metric_col2:
                        st.metric("Confidence", 
                                f"{st.session_state.current_confidence*100:.1f}%" if st.session_state.current_confidence else "N/A")

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
            text-align: center;
            margin-top: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.info("Start signing 'Hello' or 'Thank You' to see real-time transcription.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Webcam Feed")
        webrtc_ctx = webrtc_streamer(
            key="transcription",
            video_transformer_factory=TranscriptionTransformer,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

    with col2:
        st.markdown("### Transcribed Text")
        st.markdown(
            f'<div class="transcription-box">{st.session_state.transcription_text or "Waiting for transcription..."}</div>',
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
