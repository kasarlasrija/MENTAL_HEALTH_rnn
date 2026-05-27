import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download("stopwords")

stop_words = set(
    stopwords.words("english")
)

MAX_LEN = 100

st.set_page_config(
    page_title="Mental Health Monitor",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource
def load_resources():

    model = load_model(
        "mental_health_rnn_model.keras"
    )

    with open(
        "tokenizer.pkl",
        "rb"
    ) as f:

        tokenizer = pickle.load(f)

    with open(
        "label_encoder.pkl",
        "rb"
    ) as f:

        encoder = pickle.load(f)

    return (
        model,
        tokenizer,
        encoder
    )

model, tokenizer, encoder = load_resources()


def preprocess_text(text):

    text = text.lower()

    text = re.sub(
        r"[^\w\s]",
        "",
        text
    )

    words = text.split()

    words = [

        word

        for word in words

        if word not in stop_words

    ]

    return " ".join(words)


def predict_sentiment(text):

    processed = preprocess_text(
        text
    )

    sequence = tokenizer.texts_to_sequences(
        [processed]
    )

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post"
    )

    pred = model.predict(
        padded,
        verbose=0
    )[0]

    index = np.argmax(pred)

    label = encoder.inverse_transform(
        [index]
    )[0]

    confidence = pred[index]

    probabilities = dict(
        zip(
            encoder.classes_,
            pred
        )
    )

    return (
        label,
        confidence,
        probabilities,
        processed
    )


guidance = {

    "Anxiety":
    "Take deep breaths and focus on one task at a time.",

    "Depression":
    "Talk with someone you trust and take a small positive step today.",

    "Stress":
    "Take a short break and relax.",

    "Normal":
    "Keep maintaining healthy routines.",

    "Suicidal":
    "Please contact a trusted person or mental health professional.",

    "Bipolar":
    "Maintain routines and reach out for support.",

    "Personality disorder":
    "Practice emotional awareness and journaling."

}

activities = {

    "Anxiety":
    "Go for a short walk",

    "Depression":
    "Listen to calming music",

    "Stress":
    "Try meditation",

    "Normal":
    "Continue positive habits",

    "Suicidal":
    "Talk to someone trusted",

    "Bipolar":
    "Maintain sleep schedule",

    "Personality disorder":
    "Write a journal"

}


st.sidebar.title(
    "Navigation"
)

menu = st.sidebar.radio(

    "Menu",

    [

        "Home",

        "About",

        "Prediction"

    ]

)

if menu == "Home":

    st.title(
        "🧠 AI-Based Mental Health Sentiment Monitoring System"
    )

    st.subheader(
        "Emotion Detection using Simple Recurrent Neural Networks"
    )

    st.markdown("---")

    st.write(

"""
This application analyzes emotional sentiment from text using Natural Language Processing and Simple RNN.

It can:

• Predict emotional category

• Display confidence score

• Visualize emotion probabilities

• Provide emotional wellness guidance

"""
)

if menu == "About":

    st.header(
        "About Project"
    )

    st.write(

"""
### Importance of Emotional AI

Emotional AI helps identify emotional patterns in text.

### NLP Applications

- Mental wellness monitoring

- Sentiment analysis

- Counseling support systems

- Emotional intelligence systems

### Role of RNN

Simple Recurrent Neural Networks learn sequential information.

RNN remembers previous words using hidden states.

Example:

'I feel hopeless today'

The network understands emotion using word order.
"""
)

if menu == "Prediction":

    st.header(
        "Analyze Emotion"
    )

    st.write(
        "Example Inputs"
    )

    st.code(

"""I feel hopeless and tired every day

I feel nervous before exams

I am excited for tomorrow

Nobody understands me anymore"""
)

    user_input = st.text_area(

        "Enter your thoughts or feelings here...",

        height=180

    )

    if st.button(
        "Analyze Emotion"
    ):

        if user_input.strip() == "":

            st.warning(
                "Please enter text."
            )

        else:

            emotion, confidence, probs, processed = predict_sentiment(
                user_input
            )

            st.success(
                f"Emotion Detected: {emotion}"
            )

            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

            st.info(
                guidance.get(
                    emotion,
                    "Stay positive."
                )
            )

            st.success(

                "Suggested Activity: "

                +

                activities.get(
                    emotion,
                    "Take a short break"
                )

            )

            st.subheader(
                "Processed Input"
            )

            st.code(
                processed
            )

            st.subheader(
                "Emotion Probability Distribution"
            )

            chart_df = pd.DataFrame({

                "Emotion":
                list(probs.keys()),

                "Probability":
                list(probs.values())

            })

            fig = px.bar(

                chart_df,

                x="Emotion",

                y="Probability",

                title="Emotion Confidence Scores"

            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.subheader(
                "Confidence Graph"
            )

            fig2, ax = plt.subplots(
                figsize=(6,4)
            )

            ax.bar(

                chart_df["Emotion"],

                chart_df["Probability"]

            )

            plt.xticks(
                rotation=45
            )

            st.pyplot(
                fig2
            )

st.sidebar.markdown("---")

st.sidebar.write(
    "Built with TensorFlow + Streamlit"
)
