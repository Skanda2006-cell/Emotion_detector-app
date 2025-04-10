import streamlit as st
from transformers import pipeline
from datetime import datetime
import matplotlib.pyplot as plt

# Load classifier
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None, device=-1)


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

# Page config
st.set_page_config(page_title="Emotion Detector", layout="centered")

st.title("🧠 Emotion Detector 2.0 — Multi-Emotion Version")
st.write("Type how you're feeling and let the AI detect your top 3 emotions!")

text = st.text_area("Enter your message:")

if st.button("Analyze Emotion"):
    if text.strip():
        # Get all emotions
        results = classifier(text)

        # Sort and take top 3
        top_emotions = sorted(results, key=lambda x: x['score'], reverse=True)[:3]

        st.markdown("### 🧠 Top Emotions:")
        pie_labels = []
        pie_values = []

        for emotion_data in top_emotions:
            emotion = emotion_data["label"]
            score = round(emotion_data["score"] * 100, 2)
            emoji, bg_color = emotion_style.get(emotion.lower(), ("🤔", "#eeeeee"))

            # Set background color and font contrast
            text_color = "#000000" if bg_color in ["#fff9c4", "#c8e6c9", "#bbdefb", "#c5e1a5", "#eeeeee"] else "#ffffff"
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

            # Display emotion
            st.markdown(f"**{emotion}:** {score}% {emoji}")
            pie_labels.append(emotion)
            pie_values.append(score)

        # Save top 1 emotion to diary
        top1 = top_emotions[0]
        st.session_state.mood_diary.append({
            "text": text,
            "emotion": top1["label"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        # Pie chart
        st.markdown("---")
        st.subheader("📊 Top 3 Emotions Chart")
        fig, ax = plt.subplots()
        ax.pie(pie_values, labels=pie_labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

    else:
        st.warning("Please enter some text.")

# Show mood diary
if st.session_state.mood_diary:
    st.markdown("---")
    st.subheader("📝 Mood Diary")
    for entry in reversed(st.session_state.mood_diary):
        st.write(f"[{entry['timestamp']}] **{entry['emotion']}** → {entry['text']}")

    # Emotion count chart
    st.markdown("---")
    st.subheader("📈 Emotion History Chart")
    emotions = [entry['emotion'] for entry in st.session_state.mood_diary]
    counts = {e: emotions.count(e) for e in set(emotions)}

    fig, ax = plt.subplots()
    ax.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    if st.button("🧹 Reset Mood Diary"):
        st.session_state.mood_diary = []
        st.experimental_rerun()
