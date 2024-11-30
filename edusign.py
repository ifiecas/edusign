import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time

# Set up the app configuration
st.set_page_config(page_title="EduSign", layout="wide", page_icon="🖐️")

# Sidebar Navigation
st.sidebar.title("EduSign")
st.sidebar.markdown("### Empowering Communication Through Sign Language Learning")
option = st.sidebar.radio("Choose a Learning Path", ["Home", "Sign Language Tutor", "Attend Online Classes"])

# Load Machine Learning Model
MODEL_PATH = "/Users/raphael/edusign/sign_language_model.h5"
gesture_model = tf.keras.models.load_model(MODEL_PATH)

# Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Helper Function: Process Video Frame
def process_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
    return frame, result

# Home Page
if option == "Home":
    st.title("Welcome to EduSign!")
    st.subheader("Learn and Connect with Sign Language")
    st.markdown("""
    EduSign offers two amazing paths to explore:
    - 🖐️ **Sign Language Tutor**: Practice gestures and receive real-time feedback.
    - 🎓 **Attend Online Classes**: Get real-time speech-to-text transcription with webcam video.
    """)
    st.markdown("### Ready to get started? Use the sidebar to navigate!")

# Sign Language Tutor Page
elif option == "Sign Language Tutor":
    st.title("🖐️ Sign Language Tutor")
    st.markdown("**Type a word to learn its sign language and practice your gestures.**")

    # Input: Word to Learn
    word_to_learn = st.text_input("Type the word you want to learn:", "Hello")
    st.markdown(f"### Learning: **{word_to_learn.capitalize()}**")

    # Display Tutorial Video Based on Input Word
    videos = {
        "hello": "https://www.youtube.com/watch?v=iRsWS96g1B8",
        "thank you": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Replace with actual video for "Thank You"
    }

    col1, col2 = st.columns([1, 1])  # Equal-sized columns

    with col1:
        st.markdown("### Instruction Video")
        if word_to_learn.lower() in videos:
            st.video(videos[word_to_learn.lower()], format="video/mp4", start_time=0)
        else:
            st.warning("No tutorial video available for this word.")

    with col2:
        st.markdown("### Webcam Feed (Live Practice)")

        # Single checkbox to toggle webcam feed
        webcam_running = st.checkbox("Start or Stop Webcam", value=False)
        FRAME_WINDOW = st.image([])  # Placeholder for video stream
        feedback_log = []  # List to store feedback history
        last_feedback_message = None  # Track the last meaningful feedback message

        if webcam_running:
            cap = cv2.VideoCapture(0)  # Open webcam
            while webcam_running:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Unable to access the webcam.")
                    break

                # Process the frame
                frame, result = process_frame(frame)

                # Gesture Recognition
                current_feedback_message = "No hands detected. Raise your hands and ensure they are visible in the camera."
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                if result.multi_hand_landmarks:
                    landmarks = []
                    for hand_landmarks in result.multi_hand_landmarks:
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])

                    # If landmarks match the model input size, predict gesture
                    if len(landmarks) == gesture_model.input_shape[-1]:
                        prediction = gesture_model.predict(np.array(landmarks).reshape(1, -1))
                        predicted_class = np.argmax(prediction)
                        accuracy = round(np.max(prediction) * 100, 2)  # Convert to percentage

                        # Feedback for "Hello" Gesture
                        if predicted_class == 0:  # Assuming 0 is "Hello"
                            if accuracy >= 60:
                                current_feedback_message = f"Great! You signed 'Hello' correctly with {accuracy}% accuracy."
                            else:
                                current_feedback_message = f"Hands detected, but the gesture isn't clear. Try signing 'Hello' again."
                    else:
                        current_feedback_message = "Hands detected, but landmarks are incomplete. Adjust hand visibility."

                # Update feedback only on meaningful change
                if current_feedback_message != last_feedback_message:
                    last_feedback_message = current_feedback_message
                    feedback_log.insert(0, {"message": current_feedback_message, "timestamp": timestamp})

                # Display the processed video frame
                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)

                # Show the latest feedback
                if feedback_log:
                    st.markdown(f"**Latest Feedback ({feedback_log[0]['timestamp']}):** {feedback_log[0]['message']}")

                # Compile all previous feedback into one collapsible section
                if len(feedback_log) > 1:
                    with st.expander("Previous Feedback"):
                        for feedback in feedback_log[1:]:
                            st.markdown(f"- ({feedback['timestamp']}) {feedback['message']}")

                # Exit loop if the checkbox is unchecked
                if not st.session_state.get("webcam_running"):
                    break

            cap.release()  # Release webcam resources safely

# Attend Online Classes Page
elif option == "Attend Online Classes":
    st.title("🎓 Attend Online Classes")
    st.markdown("**Simulate a call with real-time transcription.**")
    st.markdown("This feature will be implemented soon.")
