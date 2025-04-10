import os
os.environ["USE_FLASH_ATTENTION"] = "0"  # Prevent SDPA/meta tensor issue

import torch
torch.set_float32_matmul_precision("high")  # Safe CPU precision

import streamlit as st
from transformers import pipeline
from datetime import datetime
import matplotlib.pyplot as plt

# Set page
st.set_page_config(page_title="Emotion Detector 2.0", layout="centered")
st.title("🧠 Emotion Detector 2.0 — Multi-Emotion Version")
st.write("Type how you're feeling and let the AI decode your top 3 emotions!")

# Load the classifier (try-except for safer load)
@st.cache_resource
def load_classifier():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True,
        device=-1
    )

classifier = load_classifier()

# Emotion emoji + color style
emotion_style = {
    "joy": ("😊", "#fff9c4"),
    "sadness": ("😢", "#bbdefb"),
    "anger": ("😠", "#ffcdd2"),
    "fear": ("😨", "#c8e6c9"),
    "love": ("❤️", "#f8bbd0"),
    "surprise": ("😲", "#d1c4e9"),
    "disgust": ("🤢", "#c5e1a5"),
}

# Mood diary
if "mood_diary" not in st.session_state:
    st.session_state.mood_diary = []

text = st.text_area("Enter your message:")

if st.button("Analyze Emotion"):
    if text.strip():
        try:
            raw_results = classifier(text)
            if isinstance(raw_results, list) and isinstance(raw_results[0], list):
                results = raw_results[0]
            else:
                results = raw_results

            top_emotions = sorted(results, key=lambda x: x['score'], reverse=True)[:3]

            # Display top 3 emotions
            for emotion_data in top_emotions:
                emotion = emotion_data["label"]
                score = round(emotion_data["score"] * 100, 2)
                emoji, bg_color = emotion_style.get(emotion.lower(), ("🤔", "#eeeeee"))
                text_color = "#000" if bg_color in ["#fff9c4", "#bbdefb", "#c8e6c9", "#c5e1a5"] else "#fff"

                st.markdown(
                    f"""
                    <div style="background-color:{bg_color}; padding:10px; border-radius:10px; color:{text_color}; margin-bottom:10px;">
                        <strong>{emotion} {emoji}</strong> — Confidence: <strong>{score}%</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Log entry
            st.session_state.mood_diary.append({
                "text": text,
                "emotions": top_emotions,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter some text.")

# Mood Diary
if st.session_state.mood_diary:
    st.markdown("---")
    st.subheader("📝 Mood Diary")

    for entry in reversed(st.session_state.mood_diary):
        st.write(f"**[{entry['timestamp']}]** → *{entry['text']}*")
        for emo in entry['emotions']:
            st.write(f"- {emo['label']} ({round(emo['score']*100, 2)}%)")

    # Chart
    st.markdown("---")
    st.subheader("📊 Emotion Chart")

    all_emotions = []
    for entry in st.session_state.mood_diary:
        all_emotions.extend([emo['label'] for emo in entry['emotions']])

    emotion_counts = {e: all_emotions.count(e) for e in set(all_emotions)}

    fig, ax = plt.subplots()
    ax.pie(emotion_counts.values(), labels=emotion_counts.keys(), autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    if st.button("🧹 Reset Mood Diary"):
        st.session_state.mood_diary = []
        st.experimental_rerun()
