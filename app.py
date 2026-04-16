import streamlit as st
from collections import Counter
import matplotlib.pyplot as plt
from googleapiclient.discovery import build
import re
from transformers import pipeline

# ⚡ Faster model
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# 🔹 YouTube comments
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

# 🎨 UI
st.set_page_config(page_title="Creator AI Analyzer", layout="wide")

st.title("🚀 Creator AI Analyzer")
st.markdown("Analyze audience sentiment & predict performance 🔥")

api_key = st.secrets["API_KEY"]

video_url = st.text_input("📺 YouTube Video Link")
max_comments = st.slider("📊 Number of comments", 50, 500, 200)

# 🚀 Analyze
if st.button("Analyze Comments"):

    video_id = extract_video_id(video_url)

    if video_id:
        comments = get_youtube_comments(video_id, api_key, max_comments)
    else:
        st.error("Invalid link ❌")
        comments = []

    if comments:
        positive = negative = neutral = 0
        all_words = []
        pos_comments = []
        neg_comments = []

        for comment in comments:
            result = sentiment_model(comment[:512])[0]
            label = result['label']

            if label == "POSITIVE":
                positive += 1
                pos_comments.append(comment)
            else:
                negative += 1
                neg_comments.append(comment)

            words = re.findall(r'\b[a-zA-Z]{4,}\b', comment.lower())
            all_words.extend(words)

        total = len(comments)

        # 📊 Metrics
        col1, col2 = st.columns(2)
        col1.metric("😊 Positive", positive)
        col2.metric("😡 Negative", negative)

        # 🔥 VIRAL PREDICTION
        st.subheader("🔥 Viral Prediction")

        positive_percent = (positive / total) * 100

        if positive_percent > 70:
            st.success("🚀 High chance of going viral!")
        elif positive_percent > 50:
            st.info("👍 Good engagement potential")
        else:
            st.error("⚠️ Low engagement expected")

        # 💬 TOP COMMENTS
        st.subheader("💬 Top Comments Insights")

        if pos_comments:
            st.write("🔥 Most Positive Comment:")
            st.success(pos_comments[0])

        if neg_comments:
            st.write("⚠️ Most Negative Comment:")
            st.error(neg_comments[0])

        # 🔥 Keywords
        st.subheader("🔥 Keywords")
        word_counts = Counter(all_words)
        common_words = word_counts.most_common(5)

        for word, count in common_words:
            st.write(f"{word}: {count}")

        # 📊 Chart
        labels = ['Positive', 'Negative']
        sizes = [positive, negative]

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%')
        st.pyplot(fig)

        # 📄 DOWNLOAD REPORT
        st.subheader("📄 Download Report")

        report = f"""
Total Comments: {total}
Positive: {positive}
Negative: {negative}
Top Keywords: {common_words}
Viral Score: {round(positive_percent,2)}%
"""

        st.download_button("📥 Download Report", report, file_name="analysis.txt")

    else:
        st.warning("No comments found")
