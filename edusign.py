import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import requests
from gtts import gTTS
import tempfile
import os
port = int(os.environ.get("PORT", 8000))
app.run(host="0.0.0.0", port=port)


# Page Configuration
st.set_page_config(page_title="EduSign@VU: Sign Language for All", layout="wide", page_icon="🖐️")

# Initialize session states
if 'webcam_running' not in st.session_state:
    st.session_state.webcam_running = False
if 'transcription_running' not in st.session_state:
    st.session_state.transcription_running = False
if 'transcription_text' not in st.session_state:
    st.session_state.transcription_text = ""
if 'usage_count' not in st.session_state:
    st.session_state.usage_count = 0  # Track user activity
if 'user_level' not in st.session_state:
    st.session_state.user_level = "Beginner"  # Initial level

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
    """Download and load the model from Azure Blob Storage."""
    model_url = "https://edusignstorage.blob.core.windows.net/model/sign_language_model_ver5.h5"
    
    try:
        # Download the model to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_file:
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            
            # Write content to the temp file
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            
            temp_file_path = temp_file.name

        # Load the model from the temporary file
        model = tf.keras.models.load_model(temp_file_path)
        return model, True

    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, False

# Load the model
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
                    
                    st.markdown(f"### EduSign AI's Feedback:\n{feedback}")
    finally:
        cap.release()

def evaluate_user_level():
    """Evaluate the user's skill level based on usage."""
    if st.session_state.usage_count < 10:
        st.session_state.user_level = "Beginner"
    elif st.session_state.usage_count < 30:
        st.session_state.user_level = "Intermediate"
    else:
        st.session_state.user_level = "Expert"


if page == "Home":
    # Full-width header image at the top
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 40px;">
            <img src="https://i.postimg.cc/wjSrs4tM/Blue-Gradient-Header-Banner-1.png" 
                 style="width: 100%; max-width: 1000px; height: auto;" alt="EduSign Header">
        </div>
        """,
        unsafe_allow_html=True
    )

    # Welcome message


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
               Whether you're a beginner or looking to advance your skills, EduSign AI provides 
               personalized guidance every step of the way.
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
                   padding: 0.5rem 1.5rem;
                   border-radius: 25px;
                   text-decoration: none;
                   font-size: 0.9rem;
                   transition: background-color 0.3s ease;
                   ">View Portfolio</a>
           </div>
       </footer>
       """,
       unsafe_allow_html=True
    )


elif page == "Sign Language Tutor":
    st.title("🖐️ EduSign - Your Sign Language Tutor")

    if not model_loaded:
        st.error("Model failed to load. Please check the URL and restart the application.")
    else:
        selected_gesture = st.selectbox("Select a word to learn:", list(gesture_classes.values()))

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Tutorial Video")
            st.markdown("""
                <div style="width: 100%; position: relative; padding-top: 56.25%;">
                    <iframe 
                        src="https://www.youtube.com/embed/Sdw7a-gQzcU"
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
                st.markdown(f"- {step}")
            st.markdown("#### Pro Tips:")
            for tip in guide.get("tips", []):
                st.markdown(f"- {tip}")
            st.markdown("#### Common Mistakes:")
            for mistake in guide.get("mistakes", []):
                st.markdown(f"- {mistake}")

        with col2:
            st.markdown("### Practice Area")
            frame_placeholder = st.empty()
            feedback_placeholder = st.empty()

            st.session_state.webcam_running = st.checkbox("Start/Stop Camera", value=st.session_state.webcam_running)

            if st.session_state.webcam_running:
                start_webcam_feed(frame_placeholder, feedback_placeholder, selected_gesture)
            else:
                st.markdown("Camera is stopped.")
            st.markdown(f"Status: {'🟢 Active' if st.session_state.webcam_running else '🔴 Inactive'}")
            st.markdown(f"Skill Level: **{st.session_state.user_level}**")



elif page == "Sign Language to Text":
    st.title("🖐️ Gesture Translator | Converting Sign Language to Text")

    if not model_loaded:
        st.error("Model failed to load. Please check the URL and restart the application.")
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

        st.info("Start waving 'Hello' or signing 'Thank You' to see real-time transcription.")

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

        st.session_state.transcription_running = st.checkbox("Start/Stop Transcription", value=st.session_state.transcription_running)

        if st.session_state.transcription_running:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                while st.session_state.transcription_running:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame, gesture, confidence = detect_gesture(frame)
                    frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

                    if gesture in ["Hello", "Thank You"] and confidence > 0.3:
                        st.session_state.transcription_text += f"{gesture} "
                        transcription_placeholder.markdown(
                            f'<div class="transcription-box">{st.session_state.transcription_text.strip()}</div>',
                            unsafe_allow_html=True
                        )
                cap.release()
        else:
            st.markdown("Transcription is stopped.")

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
                    with open(tmp_file.name, "rb") as audio_file:
                        st.audio(audio_file.read(), format="audio/mp3")




elif page == "Connect to a Mentor":
    st.title("🖐️ Connect to a Mentor")

    st.markdown(
        f"""
        ### Based on Your Learning Level: **{st.session_state.user_level}**
        Our AI-powered analysis recommends the best mentors to help you advance.
        """
    )

    mentors = {
        "Beginner": {
            "Alex": "Specializes in foundational signs and building confidence."
        },
        "Intermediate": {
            "Jordan": "Helps with fluency and transitioning to conversational signing."
        },
        "Expert": {
            "Taylor": "Expert in advanced and specialized signing techniques."
        }
    }

    st.markdown("### Recommended Mentors:")
    for mentor, description in mentors.get(st.session_state.user_level, {}).items():
        st.markdown(f"- **{mentor}**: {description}")

    st.markdown("### Select a Mentor to Schedule a Session:")
    all_mentors = {k: v for level in mentors.values() for k, v in level.items()}
    selected_mentor = st.selectbox("Choose a mentor:", list(all_mentors.keys()))

    if st.button("Schedule Session"):
        st.success(f"Session successfully scheduled with **{selected_mentor}**!")
