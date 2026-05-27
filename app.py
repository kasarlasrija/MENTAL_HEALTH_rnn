import streamlit as st
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


def build_model(num_classes):

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

            activation="relu"

        ),

        Dense(

            num_classes,

            activation="softmax"

        )

    ])

    model.compile(

        optimizer="adam",

        loss="categorical_crossentropy",

        metrics=["accuracy"]

    )

    return model


@st.cache_resource
def load_resources():

    with open(

        "tokenizer.pkl",

        "rb"

    ) as f:

        tokenizer=pickle.load(f)

    with open(

        "label_encoder.pkl",

        "rb"

    ) as f:

        encoder=pickle.load(f)

    model=build_model(

        len(
            encoder.classes_
        )

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

    confidence=float(

        np.max(pred)

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
"Please contact a trusted person or professional support immediately.",

"Bipolar":
"Maintain routines and seek support if needed.",

"Personality disorder":
"Practice emotional awareness and journaling."

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

    st.markdown("---")

    st.write("""

This application uses NLP and Simple RNN to analyze emotional sentiment.

Features:

• Emotion Detection

• Confidence Score

• Probability Visualization

• Wellness Guidance

""")


if menu=="About":

    st.header(
"About Project"
)

    st.write("""

### Emotional AI

Emotional AI detects emotional patterns from text.

### NLP Applications

• Mental Wellness

• Counseling Assistance

• Sentiment Analysis

• Emotional Monitoring

### Role of RNN

Simple RNN learns sequential information.

It remembers previous words using hidden states.

This helps understand emotional context.

""")


if menu=="Prediction":

    st.header(
"Analyze Emotion"
)

    st.code(

"""Examples:

I feel hopeless and tired every day

I feel nervous before exams

I am excited for tomorrow

Nobody understands me anymore"""

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
"Please enter text."
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

"Stay positive."

)

)

            st.success(

"Suggested Activity: "

+

activity.get(

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

title="Emotion Confidence Scores"

)

            st.plotly_chart(

fig,

use_container_width=True

)

            st.subheader(
"Confidence Graph"
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

st.sidebar.markdown("---")

st.sidebar.write(

"Built using TensorFlow + Streamlit"

)
