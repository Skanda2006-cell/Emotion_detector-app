import streamlit as st
from transformers import pipeline
from datetime import datetime
import matplotlib.pyplot as plt

# Load multi-label emotion classifier
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None, device=-1

# Emotion to emoji + color mapping
emotion_style = {
    "joy": ("😊", "#fff9c4"),
    "sadness": ("😢", "#bbdefb"),
    "anger": ("😠", "#ffcdd2"),
    "fear": ("😨", "#c8e6c9"),
    "love": ("❤️", "#f8bbd0"),
    "surprise": ("😲", "#d1c4e9"),
    "disgust": ("🤢", "#c5e1a5"),
}

# Session state for mood diary
if "mood_diary" not in st.session_state:
    st.session_state.mood_diary = []

st.set_page_config(page_title="Emotion Detector 2.0", layout="centered")

st.title("🧠 Emotion Detector 2.0 (Multi-Emotion)")
st.write("Type how you're feeling and let the AI detect top 3 emotions!")

text = st.text_area("Enter your message:")

if st.button("Analyze Emotions"):
    if text.strip():
        results = classifier(text, top_k=None)

        # Flatten results if needed
        if isinstance(results[0], list):
            results = results[0]

        # Get top 3 emotions
        top_emotions = sorted(results, key=lambda x: x['score'], reverse=True)[:3]

        # Pick the top one for styling
        main_emotion = top_emotions[0]['label']
        emoji, bg_color = emotion_style.get(main_emotion.lower(), ("🤔", "#eeeeee"))
        text_color = "#000000" if bg_color in ["#fff9c4", "#c8e6c9", "#bbdefb", "#c5e1a5", "#eeeeee"] else "#ffffff"

        # Styling
        st.markdown(
            f"""
            <style>
            div[data-testid="stApp"] {{
                background-color: {bg_color};
                color: {text_color};
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

        # Show emotions
        for item in top_emotions:
            emotion = item["label"]
            score = round(item["score"] * 100, 2)
            emo_icon = emotion_style.get(emotion.lower(), ("🤔",))[0]
            st.markdown(f"### {emotion}: {emo_icon} — {score}%")

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
        emo_display = ", ".join(f"{e['label']} ({round(e['score']*100)}%)" for e in entry['emotions'])
        st.write(f"[{entry['timestamp']}] → **{emo_display}** → _{entry['text']}_")

    # Pie Chart
    st.markdown("---")
    st.subheader("📊 Top Emotions Chart")
    all_emotions = []
    for entry in st.session_state.mood_diary:
        all_emotions.extend([e['label'] for e in entry['emotions']])
    counts = {e: all_emotions.count(e) for e in set(all_emotions)}

    fig, ax = plt.subplots()
    ax.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    if st.button("🧹 Reset Mood Diary"):
        st.session_state.mood_diary = []
        st.experimental_rerun()
