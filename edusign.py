import streamlit as st

# Initialize session state at the top
st.session_state.setdefault("debug_mode", False)
st.session_state.setdefault("target_gesture", "None")  # Default to "None"
st.session_state.setdefault("transcription_text", "")
st.session_state.setdefault("usage_count", 0)
st.session_state.setdefault("user_level", "Beginner")
st.session_state.setdefault("current_prediction", None)
st.session_state.setdefault("current_confidence", None)
st.session_state.setdefault("feedback_text", "")
st.session_state.setdefault("last_transcribed_gesture", None)
st.session_state.setdefault("real_time_gesture", "")
st.session_state.setdefault("real_time_confidence", None)
st.session_state.setdefault("last_detection_time", 0.0)

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

st.set_page_config(page_title="EduSign@VU: Sign Language for All", layout="wide", page_icon="🖐️")

CONFIDENCE_THRESHOLD = 0.30
MIN_CONFIDENCE = 0.20
TRANSCRIPTION_THRESHOLD = 0.30
DISPLAY_THRESHOLD = 0.20

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
        if st.session_state.get("debug_mode", False):
            print("Model loaded successfully!")
        return model, True
    except Exception as e:
        if st.session_state.get("debug_mode", False):
            print(f"Model loading error: {e}")
        st.error(f"Failed to load model: {e}")
        return None, False

gesture_model, model_loaded = load_model()

gesture_classes = {0: "Hello", 1: "Thank You", 2: "Yes", 3: "No"}

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

def get_color_for_confidence(confidence):
    if confidence >= CONFIDENCE_THRESHOLD:
        return (0, 255, 0)
    elif confidence >= MIN_CONFIDENCE:
        return (0, 255, 255)
    else:
        return (0, 0, 255)

def generate_feedback(prediction, confidence, target_gesture):
    if prediction is None:
        return "No hand detected. Please show your hand to the camera.", "info"

    if not target_gesture or target_gesture not in learning_guides:
        return "No target gesture selected or invalid target gesture.", "info"

    feedback = f"Detected: {prediction} ({confidence:.1%})\n\n"

    if confidence <= MIN_CONFIDENCE:
        feedback += f"Tips for '{target_gesture}':\n"
        for tip in learning_guides[target_gesture]["tips"]:
            feedback += f"• {tip}\n"
        return feedback, "error"
    elif prediction != target_gesture:
        feedback += f"Trying to learn: '{target_gesture}'\n\n"
        feedback += "Remember these steps:\n"
        for step in learning_guides[target_gesture]["steps"]:
            feedback += f"• {step}\n"
        return feedback, "warning"
    else:
        if confidence < CONFIDENCE_THRESHOLD:
            feedback += "Getting better! Common mistakes to avoid:\n"
            for mistake in learning_guides[target_gesture]["mistakes"]:
                feedback += f"• {mistake}\n"
            return feedback, "warning"
        else:
            feedback += "Great job! Keep practicing to maintain this level!"
            return feedback, "success"


class GestureTutorProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        target_gesture = st.session_state.get("target_gesture", "None")
        cv2.putText(img, f"Target: {target_gesture}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

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

                if x_min < x_max and y_min < y_max:
                    hand_img = rgb_frame[y_min:y_max, x_min:x_max]
                    hand_img = cv2.resize(hand_img, (224, 224))
                    hand_img = hand_img / 255.0

                    try:
                        pred = gesture_model.predict(np.expand_dims(hand_img, axis=0), verbose=0)
                        prediction = gesture_classes.get(np.argmax(pred))
                        confidence = float(np.max(pred))

                        st.session_state["current_prediction"] = prediction
                        st.session_state["current_confidence"] = confidence

                        feedback_text, feedback_type = generate_feedback(
                            prediction, confidence, st.session_state.get("target_gesture", "None")
                        )
                        st.session_state["feedback_text"] = feedback_text

                        color = get_color_for_confidence(confidence)

                        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

                        label = f"Detected: {prediction} ({confidence:.1%})"
                        (text_width, text_height), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        cv2.rectangle(img,
                                      (x_min, y_min - text_height - 10),
                                      (x_min + text_width, y_min),
                                      (255, 255, 255),
                                      -1)
                        cv2.putText(img, label, (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                        # Transcribe at ≥20% confidence
                        if confidence >= TRANSCRIPTION_THRESHOLD and prediction is not None:
                            st.session_state["transcription_text"] += f"{prediction} "

                    except Exception as e:
                        print(f"Prediction error: {e}")
        else:
            st.session_state["current_prediction"] = None
            st.session_state["current_confidence"] = None
            st.session_state["feedback_text"] = "No hand detected. Please show your hand to the camera."

        return av.VideoFrame.from_ndarray(img, format="bgr24")


class TranscriptionProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.last_prediction = None
        self.last_prediction_time = 0
        self.cooldown = 1.0

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
                x_min, x_max = max(0, x_min - padding), min(w, x_max + padding)
                y_min, y_max = max(0, y_min - padding), min(h, y_max + padding)

                if current_time - self.last_prediction_time >= self.cooldown:
                    hand_img = rgb_frame[y_min:y_max, x_min:x_max]
                    hand_img = cv2.resize(hand_img, (224, 224))
                    hand_img = hand_img / 255.0

                    try:
                        pred = gesture_model.predict(np.expand_dims(hand_img, axis=0), verbose=0)
                        prediction = gesture_classes.get(np.argmax(pred))
                        confidence = float(np.max(pred))

                        label = f"{prediction}: {confidence:.1%}"
                        cv2.putText(img, label, (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        if confidence > TRANSCRIPTION_THRESHOLD and prediction != self.last_prediction:
                            st.session_state["transcription_text"] += f"{prediction} "
                            self.last_prediction = prediction
                            self.last_prediction_time = current_time

                    except Exception as e:
                        print(f"Prediction error: {e}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")


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

if page == "Home":
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 40px;">
            <img src="https://i.imghippo.com/files/CfXN6612ESw.png" alt="" border="0" style="width: 100%; max-width: 1000px; height: auto;" alt="EduSign Header">
        </div>
        
        <div style="text-align: center; background-color: #f8f9fa; padding: 2.5rem 0; border-radius: 10px;">
            <h2 style="color: #0f2f76; font-size: 1.8rem; line-height: 1.5; margin: 0 auto; max-width: 800px; padding: 0 20px;">
                EduSign AI is an innovative educational platform developed in partnership with 
                <span style="color: #0f2f76; font-weight: 700;">Victoria University</span>.
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns(3)
    features = [
        ("🎯", "Smart Learning", "AI-powered real-time feedback"),
        ("📱", "Instant Translation", "Sign-to-text conversion"),
        ("🤝", "Expert Guidance", "Learn from certified mentors")
    ]
    
    for col, (emoji, title, desc) in zip([col1, col2, col3], features):
        with col:
            st.markdown(
                f"""
                <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); text-align: center; height: 100%;">
                    <span style="font-size: 2.5rem;">{emoji}</span>
                    <h4 style="color: #0f2f76; margin: 1rem 0;">{title}</h4>
                    <p style="color: #666;">{desc}</p>
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
        st.session_state["target_gesture"] = selected_gesture

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
            st.markdown(f"#### Steps for '{selected_gesture}':")
            for step in guide.get("steps", []):
                st.markdown(f"- {step}")

        with col2:
            st.markdown("### Practice Area")
            webrtc_ctx = webrtc_streamer(
                key="gesture-tutor",
                video_processor_factory=GestureTutorProcessor,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
                async_processing=False
            )

            if webrtc_ctx.state.playing:
                st.markdown("### Current Detection")
                cols = st.columns([1, 1])
                with cols[0]:
                    detected_gesture = st.session_state.get("current_prediction", "None")
                    st.metric("Detected Gesture", detected_gesture)
                with cols[1]:
                    confidence = st.session_state.get("current_confidence", None)
                    if confidence is not None:
                        color = "red" if confidence < CONFIDENCE_THRESHOLD else "green"
                        st.markdown(f'<p style="color: {color}; font-size: 1.2em;">Confidence: {confidence:.1%}</p>', 
                                    unsafe_allow_html=True)

                st.markdown("### EduSign AI's Feedback")
                feedback = st.session_state.get("feedback_text", "")
                if feedback:
                    if "Great job" in feedback:
                        st.success(feedback)
                    elif "Getting better" in feedback or "Trying to learn" in feedback:
                        st.warning(feedback)
                    elif "No hand detected" in feedback:
                        st.info(feedback)
                    else:
                        st.warning(feedback)
                else:
                    st.info("Show your hand to get feedback")

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
    </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Webcam Feed")
        webrtc_ctx = webrtc_streamer(
            key="transcription",
            video_processor_factory=TranscriptionProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
            async_processing=False
        )

    with col2:
        st.markdown("### Transcribed Text")
        transcription = st.session_state.get("transcription_text", "Waiting for signs...")
        st.markdown(
            f'<div class="transcription-box">{transcription}</div>',
            unsafe_allow_html=True
        )

        button_col1, button_col2 = st.columns(2)
        with button_col1:
            if st.button("Clear Text"):
                st.session_state["transcription_text"] = ""
                st.experimental_rerun()
            
            if st.button("Download Text"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
                    tmp_file.write(st.session_state["transcription_text"].encode())
                    st.download_button("Download", tmp_file.name, "transcription.txt")

        with button_col2:
            if st.button("Listen to Text"):
                text_to_speak = st.session_state["transcription_text"].strip()
                if text_to_speak:
                    tts = gTTS(text_to_speak)
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

    user_level = st.session_state.get("user_level", "Beginner")
    st.markdown(
        f"""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h3>Your Current Level: <span style="color: #0f2f76;">{user_level}</span></h3>
            <p>Usage Count: {st.session_state.get("usage_count",0)} sessions</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    level_mentors = mentors.get(user_level, {})
    
    # Recommend a mentor based on user's learning level
    # For simplicity, pick the first mentor in the dictionary as the recommended one
    recommended_mentor = next(iter(level_mentors.keys()))
    recommended_mentor_description = level_mentors.get(recommended_mentor, "")

    st.markdown(
        f"""
        <div style="background-color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h3 style="color: #0f2f76;">Recommended Mentor for Your Level</h3>
            <p style="font-size: 1.1rem;">
                Based on your current learning path <span style="color: #0f2f76; font-weight: 600;">({user_level})</span>,
                we recommend connecting with <span style="color: #0f2f76; font-weight: 600;">{recommended_mentor}</span>.
            </p>
            <p style="font-size: 1.1rem;">{recommended_mentor_description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # User can still choose a different mentor if desired
    selected_mentor = st.selectbox("Choose your mentor:", list(level_mentors.keys()), index=0)

    # Use a calendar input for selecting the date
    preferred_date = st.date_input("Select Preferred Date:")

    # Time selection from a dropdown
    preferred_time = st.selectbox("Select Preferred Time:", ["9:00 AM", "10:00 AM", "11:00 AM", "2:00 PM", "3:00 PM", "4:00 PM"])

    if st.button("Schedule Session"):
        st.session_state["usage_count"] = st.session_state.get("usage_count", 0) + 1
        st.success(f"✅ Session scheduled with {selected_mentor} on {preferred_date} at {preferred_time}")
        st.balloons()


# Add a universal footer for all pages
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
