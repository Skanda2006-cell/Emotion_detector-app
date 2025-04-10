import streamlit as st
from transformers import pipeline
from datetime import datetime
import matplotlib.pyplot as plt

# Emotion to emoji and color mapping
emotion_style = {
    "joy": ("😊", "#fff9c4"),
    "sadness": ("😢", "#bbdefb"),
    "anger": ("😠", "#ffcdd2"),
    "fear": ("😨", "#c8e6c9"),
    "love": ("❤️", "#f8bbd0"),
    "surprise": ("😲", "#d1c4e9"),
    "disgust": ("🤢", "#c5e1a5"),
}

# Initialize model (CPU only)
classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    device=-1  # use CPU
)

# Set page
st.set_page_config(page_title="Emotion Detector 2.0", layout="centered")
st.title("🧠 Emotion Detector 2.0 — Multi-Emotion Version")
st.write("Enter your thoughts, and I'll detect the top 3 emotions!")

# Mood diary state
if "mood_diary" not in st.session_state:
    st.session_state.mood_diary = []

# Input
text = st.text_area("How are you feeling?")

if st.button("🔍 Analyze"):
    if text.strip():
        results = classifier(text)
        top_emotions = sorted(results, key=lambda x: x['score'], reverse=True)[:3]

        for emotion_data in top_emotions:
            emotion = emotion_data["label"]
            score = round(emotion_data["score"] * 100, 2)
            emoji, bg_color = emotion_style.get(emotion.lower(), ("🤔", "#eeeeee"))
            text_color = "#000000" if bg_color in ["#fff9c4", "#c8e6c9", "#bbdefb", "#c5e1a5", "#eeeeee"] else "#ffffff"

            st.markdown(
                f"""
                <div style='background-color:{bg_color}; padding:10px; border-radius:10px; color:{text_color}; margin-bottom:10px;'>
                <b>{emotion}</b> {emoji} — <b>{score}%</b>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Save to mood diary
        st.session_state.mood_diary.append({
            "text": text,
            "emotions": top_emotions,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    else:
        st.warning("Please enter some text.")

# Mood Diary
if st.session_state.mood_diary:
    st.markdown("---")
    st.subheader("📝 Mood Diary")
    for entry in reversed(st.session_state.mood_diary):
        st.write(f"**[{entry['timestamp']}]** {entry['text']}")
        for e in entry["emotions"]:
            st.write(f"• {e['label']} — {round(e['score'] * 100, 2)}%")

    # Emotion Chart
    st.markdown("---")
    st.subheader("📊 Emotion Chart (Top Emotions)")

    # Flatten and count
    emotion_counter = {}
    for entry in st.session_state.mood_diary:
        for e in entry["emotions"]:
            emotion_counter[e["label"]] = emotion_counter.get(e["label"], 0) + 1

    fig, ax = plt.subplots()
    ax.pie(emotion_counter.values(), labels=emotion_counter.keys(), autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    if st.button("🧹 Reset Mood Diary"):
        st.session_state.mood_diary = []
        st.experimental_rerun()
