import streamlit as st
from collections import Counter
import matplotlib.pyplot as plt
from googleapiclient.discovery import build
import re
from transformers import pipeline

# 🔥 Load AI model
sentiment_model = pipeline("sentiment-analysis")

# 🔹 Get YouTube comments
def get_youtube_comments(video_id, api_key, max_comments=200):
    youtube = build('youtube', 'v3', developerKey=api_key)

    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=50,
            pageToken=next_page_token
        )

        response = request.execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    return comments[:max_comments]

# 🔹 Extract video ID
def extract_video_id(url):
    patterns = [
        r"v=([^&]+)",
        r"youtu\.be/([^?&]+)",
        r"shorts/([^?&]+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


# 🎨 UI CONFIG
st.set_page_config(page_title="Creator AI Analyzer", layout="wide")

st.title("🚀 Creator AI Analyzer")
st.markdown("### Understand your audience. Improve your content. Grow faster 🔥")

st.divider()

# 🔐 Hidden API key
api_key = st.secrets["API_KEY"]

# 📥 INPUT SECTION
with st.container():
    st.subheader("📥 Input")

    col1, col2 = st.columns(2)

    with col1:
        video_url = st.text_input("📺 YouTube Video Link")

    with col2:
        max_comments = st.slider("📊 Number of comments", 50, 500, 200)

    comments_input = st.text_area("✍️ Or paste comments manually")

# 🚀 ANALYZE
if st.button("Analyze Comments"):

    if video_url:
        video_id = extract_video_id(video_url)

        if video_id:
            comments = get_youtube_comments(video_id, api_key, max_comments)
        else:
            st.error("Invalid YouTube link ❌")
            comments = []
    else:
        comments = comments_input.split("\n")

    if comments:
        positive = negative = neutral = 0
        all_words = []

        for comment in comments:
            result = sentiment_model(comment[:512])[0]
            label = result['label']

            if label == "POSITIVE":
                positive += 1
            elif label == "NEGATIVE":
                negative += 1
            else:
                neutral += 1

            words = re.findall(r'\b[a-zA-Z]{4,}\b', comment.lower())
            all_words.extend(words)

        total = len(comments)

        # 📊 RESULTS UI
        st.subheader("📊 Analysis Results")

        col1, col2, col3 = st.columns(3)
        col1.metric("😊 Positive", positive)
        col2.metric("😡 Negative", negative)
        col3.metric("😐 Neutral", neutral)

        # 📊 PIE CHART
        labels = ['Positive', 'Negative', 'Neutral']
        sizes = [positive, negative, neutral]

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%')
        ax1.axis('equal')
        st.pyplot(fig1)

        # 🔥 KEYWORDS
        st.subheader("🔥 Top Keywords")
        word_counts = Counter(all_words)
        common_words = word_counts.most_common(5)

        for word, count in common_words:
            st.write(f"{word} : {count}")

        # 📊 BAR CHART
        words = [w for w, c in common_words]
        counts = [c for w, c in common_words]

        fig2, ax2 = plt.subplots()
        ax2.bar(words, counts)
        st.pyplot(fig2)

        # 💡 SMART INSIGHTS
        st.subheader("💡 Smart Insights")

        if positive > negative:
            st.success("🔥 Your audience LOVES this content style. Keep doing similar videos!")
        elif negative > positive:
            st.error("⚠️ Audience is not satisfied. Improve quality or topic selection.")
        else:
            st.info("🤔 Mixed reactions. Try experimenting with new ideas.")

        # 🚀 CONTENT STRATEGY
        st.subheader("🚀 Content Strategy Suggestions")

        top_words = [word for word, count in common_words]

        if top_words:
            st.write(f"📌 Audience is talking about: {', '.join(top_words)}")

        if any("edit" in word for word in top_words):
            st.write("🎬 Improve or maintain strong editing style")

        if any("audio" in word for word in top_words):
            st.write("🔊 Improve audio clarity")

        if any("boring" in word for word in top_words):
            st.write("⚡ Make content more engaging and fast-paced")

        if any("good" in word or "great" in word for word in top_words):
            st.write("🔥 Audience is impressed — repeat this format")

        # 🎯 FINAL AI RECOMMENDATION
        if positive > 70:
            st.success("🚀 High engagement content — scale this style!")
        elif negative > 50:
            st.error("❗ Improve content quality immediately")
        else:
            st.info("👉 Try new formats or topics to grow faster")

    else:
        st.warning("No comments found")


