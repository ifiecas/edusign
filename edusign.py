import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from datetime import datetime

# Set up the app configuration
st.set_page_config(page_title="EduSign", layout="wide", page_icon="🖐️")

# Sidebar Navigation
st.sidebar.title("EduSign")
st.sidebar.markdown("### Empowering Communication Through Sign Language Learning")
option = st.sidebar.radio(
    "Choose a Learning Path",
    ["Home", "Sign Language Tutor", "Attend Online Classes"]
)

# Load Machine Learning Model with Error Handling
MODEL_PATH = "/Users/raphael/signlanguage_tutor/sign_language_model_ver4.h5"
try:
    gesture_model = tf.keras.models.load_model(MODEL_PATH)
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load the model: {e}")
    model_loaded = False

# Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Gesture Classes Mapping
gesture_classes = {0: "Hello", 1: "Thank You", 2: "Yes", 3: "No"}  # Defined gesture classes

# Helper Function: Extract Hand Region
def extract_hand_region(frame, hand_landmarks):
    h, w, _ = frame.shape
    x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
    x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
    y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
    y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

    # Add padding to ensure hand is fully captured
    padding = 20
    x_min = max(x_min - padding, 0)
    x_max = min(x_max + padding, w)
    y_min = max(y_min - padding, 0)
    y_max = min(y_max + padding, h)

    # Crop the hand region
    hand_img = frame[y_min:y_max, x_min:x_max]
    return hand_img

# Helper Function: Process Video Frame
def process_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    predicted_class = None
    confidence = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Preprocess hand region
            hand_img = extract_hand_region(frame_rgb, hand_landmarks)
            if hand_img.size > 0:
                hand_img_resized = cv2.resize(hand_img, (224, 224))
                hand_img_normalized = hand_img_resized / 255.0  # Normalize pixel values
                input_data = np.expand_dims(hand_img_normalized, axis=0)

                # Predict gesture
                prediction = gesture_model.predict(input_data)
                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction)

    return frame, predicted_class, confidence

# Webcam Feed Function
def start_webcam(FRAME_WINDOW, feedback_placeholder):
    cap = cv2.VideoCapture(0)  # Start the webcam
    if not cap.isOpened():
        st.error("Unable to access the webcam. Please ensure your webcam is connected and accessible.")
        return

    while st.session_state.webcam_running:
        ret, frame = cap.read()
        if not ret:
            st.warning("Unable to read frames from the webcam. Exiting...")
            break

        # Process the frame
        frame, predicted_class, confidence = process_frame(frame)

        # Prepare feedback
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        gesture = "Unknown Gesture" if predicted_class is None else gesture_classes.get(predicted_class, "Unknown Gesture")
        confidence_percentage = confidence * 100 if confidence else 0

        # Display the frame in the Streamlit app
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)

        # Update real-time feedback in the placeholder
        with feedback_placeholder.container():
            st.markdown(f"### Real-Time Feedback")
            st.markdown(f"**Time:** {current_time}")
            st.markdown(f"**Gesture:** {gesture}")
            st.markdown(f"**Accuracy:** {confidence_percentage:.2f}%")
            if confidence and confidence < 0.4:
                st.warning("Low confidence detected. Try the following tips:")
                st.markdown("- Ensure your hand is visible to the camera.")
                st.markdown("- Avoid overlapping fingers.")
                st.markdown("- Hold your hand steady.")
            elif confidence:
                st.success("Great job! Your gesture is clear!")

    cap.release()  # Safely release the webcam

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
    st.markdown("**Select a word to learn its sign language and practice your gestures.**")

    # Check if the model was loaded successfully
    if not model_loaded:
        st.error("The gesture recognition model could not be loaded. Please check the model file and restart the application.")
    else:
        st.success("The gesture recognition model was loaded successfully!")

        # Dropdown for selecting a word to learn
        word_to_learn = st.selectbox("Select the word you want to learn:", ["Hello", "Thank You", "Yes", "No"])
        st.markdown(f"### Learning: **{word_to_learn}**")

        # Display Tutorial Video Based on Selected Word
        videos = {
            "Hello": "https://www.youtube.com/watch?v=iRsWS96g1B8",  # Replace with actual video for "Hello"
            "Thank You": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Replace with actual video for "Thank You"
            "Yes": "https://www.youtube.com/watch?v=exampleYes",  # Replace with actual video for "Yes"
            "No": "https://www.youtube.com/watch?v=exampleNo"  # Replace with actual video for "No"
        }

        col1, col2 = st.columns([1, 1])  # Equal-sized columns

        with col1:
            st.markdown("### Instruction Video")
            st.video(videos[word_to_learn], format="video/mp4", start_time=0)

        with col2:
            st.markdown("### Webcam Feed (Live Practice)")

            # Webcam State Management
            if "webcam_running" not in st.session_state:
                st.session_state.webcam_running = False

            # Webcam Toggle Button
            webcam_running = st.checkbox("Start Webcam", value=st.session_state.webcam_running)
            st.session_state.webcam_running = webcam_running

            FRAME_WINDOW = st.image([])  # Placeholder for video stream
            feedback_placeholder = st.empty()  # Placeholder for feedback

            if st.session_state.webcam_running:
                start_webcam(FRAME_WINDOW, feedback_placeholder)

# Attend Online Classes Page
elif option == "Attend Online Classes":
    st.title("🎓 Attend Online Classes")
    st.markdown("**Simulate a call with real-time transcription.**")
    st.markdown("This feature will be implemented soon.")
