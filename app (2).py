import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Load Model and Preprocessing Objects
# --------------------------------------------------

model = tf.keras.models.load_model("rnn_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

MAX_LEN = 100

# --------------------------------------------------
# Text Preprocessing Function
# --------------------------------------------------

def preprocess_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding='post',
        truncating='post'
    )
    return padded

# --------------------------------------------------
# Guidance Messages
# --------------------------------------------------

guidance = {
    "Anxiety":
        "Take a short break and talk with someone you trust.",
    "Depression":
        "Try a small positive activity such as a walk or journaling.",
    "Stress":
        "Practice deep breathing and organize one task at a time.",
    "Suicidal":
        "Please seek immediate support from a mental health professional or trusted person.",
    "Normal":
        "Keep maintaining healthy habits and positive routines.",
    "Bipolar":
        "Maintain regular sleep patterns and follow professional guidance.",
    "Personality Disorder":
        "Focus on self-awareness and professional emotional support."
}

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Mental Health Sentiment Monitoring",
    layout="wide"
)

# ==================================================
# SECTION 1 — HEADER
# ==================================================

st.title("AI-Based Mental Health Sentiment Monitoring System")
st.subheader("Emotion Detection using Simple Recurrent Neural Networks")

# ==================================================
# SECTION 2 — ABOUT PROJECT
# ==================================================

st.markdown("---")
st.header("About the Project")

st.write("""
### Importance of Emotional AI
Emotional AI helps machines understand human emotions from text, speech, and behavior. It supports early emotional assessment and mental wellness monitoring.

### NLP Applications
Natural Language Processing (NLP) enables sentiment analysis, emotion detection, chatbots, virtual assistants, recommendation systems, and healthcare applications.

### Role of RNN in Sequence Learning
Simple Recurrent Neural Networks (RNNs) process sequential text data by remembering previous information in a sequence, making them effective for emotion and sentiment classification.
""")

# ==================================================
# SECTION 3 — USER INPUT
# ==================================================

st.markdown("---")
st.header("Enter Text for Analysis")

st.info("""
Sample Sentences:
• I feel nervous and worried about my future.
• I am very happy today.
• Nothing seems meaningful anymore.
• I feel stressed because of my workload.
""")

user_text = st.text_area(
    "Your Text",
    height=180,
    placeholder="Enter your thoughts or feelings here..."
)

# ==================================================
# SECTION 4 — PREDICTION BUTTON
# ==================================================

predict_btn = st.button("Analyze Emotion")

# ==================================================
# SECTION 5, 6, 7
# ==================================================

if predict_btn:

    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:

        processed_text = preprocess_text(user_text)

        prediction = model.predict(processed_text, verbose=0)

        predicted_index = np.argmax(prediction)
        confidence = float(np.max(prediction))

        predicted_emotion = label_encoder.inverse_transform(
            [predicted_index]
        )[0]

        # ------------------------------------------
        # SECTION 5 — Prediction Output
        # ------------------------------------------

        st.markdown("---")
        st.header("Prediction Result")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Emotion Detected",
            predicted_emotion
        )

        col2.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )

        col3.metric(
            "Emotional Status",
            predicted_emotion
        )

        # ------------------------------------------
        # SECTION 6 — Visualization
        # ------------------------------------------

        st.markdown("---")
        st.header("Visualization Area")

        emotions = label_encoder.classes_
        probabilities = prediction[0]

        chart_df = pd.DataFrame({
            "Emotion": emotions,
            "Probability": probabilities
        })

        st.bar_chart(
            chart_df.set_index("Emotion")
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(emotions, probabilities, marker="o")
        ax.set_title("Sentiment Confidence Graph")
        ax.set_ylabel("Probability")
        ax.set_xlabel("Emotion")

        st.pyplot(fig)

        # ------------------------------------------
        # SECTION 7 — Emotional Guidance
        # ------------------------------------------

        st.markdown("---")
        st.header("Emotional Guidance")

        message = guidance.get(
            predicted_emotion,
            "Stay mindful and take care of your emotional well-being."
        )

        st.success(message)

        st.write("### Positive Activity Suggestion")
        st.write("✔ Take a short walk")
        st.write("✔ Practice mindfulness")
        st.write("✔ Listen to calming music")
        st.write("✔ Talk with a trusted friend")

        st.write("### Wellness Tips")
        st.write("""
        - Maintain healthy sleep habits.
        - Stay hydrated.
        - Exercise regularly.
        - Limit excessive stress exposure.
        - Seek professional help if needed.
        """)
