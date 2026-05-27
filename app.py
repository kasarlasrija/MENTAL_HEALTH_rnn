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
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    SimpleRNN,
    Dense,
    Dropout
)

nltk.download(
    "stopwords"
)

stop_words=set(
    stopwords.words(
        "english"
    )
)

MAX_LEN=100
VOCAB_SIZE=10000

st.set_page_config(

    page_title=
    "Mental Health Monitor",

    page_icon="🧠",

    layout="wide"

)

def build_model(
    num_classes
):

    model=Sequential([

        Embedding(

            VOCAB_SIZE,

            128,

            input_length=MAX_LEN

        ),

        SimpleRNN(

            128,

            return_sequences=True

        ),

        Dropout(
            0.3
        ),

        SimpleRNN(
            64
        ),

        Dropout(
            0.3
        ),

        Dense(

            64,

            activation='relu'

        ),

        Dense(

            num_classes,

            activation='softmax'

        )

    ])

    return model


@st.cache_resource
def load_resources():

    with open(

        "tokenizer.pkl",

        "rb"

    ) as f:

        tokenizer=pickle.load(
            f
        )

    with open(

        "label_encoder.pkl",

        "rb"

    ) as f:

        encoder=pickle.load(
            f
        )

    num_classes=len(
        encoder.classes_
    )

    model=build_model(
        num_classes
    )

    model.load_weights(

        "mental_health_weights.weights.h5"

    )

    return (

        model,

        tokenizer,

        encoder

    )


model,tokenizer,encoder=load_resources()


def preprocess_text(text):

    text=text.lower()

    text=re.sub(

        r"[^\w\s]",

        "",

        text

    )

    words=text.split()

    words=[

        word

        for word in words

        if word not in stop_words

    ]

    return " ".join(
        words
    )


def predict_sentiment(text):

    processed=preprocess_text(
        text
    )

    seq=tokenizer.texts_to_sequences(

        [processed]

    )

    padded=pad_sequences(

        seq,

        maxlen=MAX_LEN,

        padding="post"

    )

    pred=model.predict(

        padded,

        verbose=0

    )[0]

    index=np.argmax(
        pred
    )

    emotion=encoder.inverse_transform(

        [index]

    )[0]

    confidence=np.max(
        pred
    )

    probs=dict(

        zip(

            encoder.classes_,

            pred

        )

    )

    return (

        emotion,

        confidence,

        probs,

        processed

    )


guidance={

"Anxiety":
"Take deep breaths and focus on one task at a time.",

"Depression":
"Talk with someone you trust and take a small positive step today.",

"Stress":
"Take a short break and relax.",

"Normal":
"Keep maintaining healthy routines.",

"Suicidal":
"Please contact trusted support immediately.",

"Bipolar":
"Maintain routines and seek support.",

"Personality disorder":
"Practice emotional awareness."

}

activity={

"Anxiety":
"Go for a short walk",

"Depression":
"Listen to calming music",

"Stress":
"Try meditation",

"Normal":
"Continue positive habits",

"Suicidal":
"Talk with someone trusted",

"Bipolar":
"Maintain sleep schedule",

"Personality disorder":
"Write a journal"

}


st.sidebar.title(
"Navigation"
)

menu=st.sidebar.radio(

"Menu",

[

"Home",

"About",

"Prediction"

]

)


if menu=="Home":

    st.title(

"🧠 AI-Based Mental Health Sentiment Monitoring System"

)

    st.subheader(

"Emotion Detection using Simple Recurrent Neural Networks"

)

    st.write("""

Analyze emotional sentiment patterns using NLP and RNN.

• Emotion Detection

• Probability Visualization

• Confidence Scores

• Wellness Guidance

""")


if menu=="About":

    st.header(
"About Project"
)

    st.write("""

Emotional AI detects emotional patterns from text.

NLP Applications:

• Sentiment Analysis

• Mental Wellness

• Counseling Systems

Simple RNN learns sequence information using hidden states.

Previous words influence future understanding.

""")


if menu=="Prediction":

    st.header(
"Analyze Emotion"
)

    text=st.text_area(

"Enter your thoughts or feelings here...",

height=180

)

    if st.button(

"Analyze Emotion"

):

        if text.strip()=="":

            st.warning(
"Enter text"
)

        else:

            emotion,confidence,probs,processed=\

            predict_sentiment(
                text
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

"Stay positive"

)

)

            st.success(

"Suggested Activity: "

+

activity.get(

emotion,

"Take a break"

)

)

            st.code(
processed
)

            df=pd.DataFrame({

"Emotion":
list(probs.keys()),

"Probability":
list(probs.values())

})

            fig=px.bar(

df,

x="Emotion",

y="Probability",

title="Emotion Probability"

)

            st.plotly_chart(

fig,

use_container_width=True

)

            fig2,ax=\

            plt.subplots()

            ax.bar(

df["Emotion"],

df["Probability"]

)

            plt.xticks(
rotation=45
)

            st.pyplot(
fig2
)

st.sidebar.write(

"Built using TensorFlow + Streamlit"

)