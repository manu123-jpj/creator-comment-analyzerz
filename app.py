import streamlit as st
from collections import Counter
import matplotlib.pyplot as plt
from googleapiclient.discovery import build
import re
from transformers import pipeline

# ⚡ Fast AI model
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

# 🎨 PAGE CONFIG
st.set_page_config(page_title="Creator AI Dashboard", layout="wide")

# 🎨 SIDEBAR
st.sidebar.title("🚀 Creator AI")
st.sidebar.markdown("### Dashboard Menu")
st.sidebar.info("Analyze audience & grow faster")

# 🔐 API key
api_key = st.secrets["API_KEY"]

# 🎯 HEADER
st.markdown("""
<h1 style='text-align: center;'>🔥 Comment Analyzer Pro</h1>
<p style='text-align: center; font-size:18px;'>AI-powered insights for smarter content creation</p>
<p style='text-align: center; font-size:14px; color:gray;'>™ Manu's</p>
""", unsafe_allow_html=True)

st.divider()

# 📥 INPUT SECTION (CARD STYLE)
with st.container():
    st.markdown("### 📥 Input")

    col1, col2 = st.columns([2,1])

    with col1:
        video_url = st.text_input("📺 YouTube Video Link")

    with col2:
        max_comments = st.slider("📊 Comments", 50, 500, 200)

# 🚀 ANALYZE BUTTON
if st.button("🚀 Analyze Comments"):

    video_id = extract_video_id(video_url)

    if video_id:
        comments = get_youtube_comments(video_id, api_key, max_comments)
    else:
        st.error("Invalid YouTube link ❌")
        comments = []

    if comments:
        positive = negative = 0
        pos_comments = []
        neg_comments = []
        all_words = []

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
        positive_percent = (positive / total) * 100

        # 📊 METRICS CARDS
        st.markdown("## 📊 Performance Overview")
        col1, col2 = st.columns(2)
        col1.metric("😊 Positive", positive)
        col2.metric("😡 Negative", negative)

        # 🔥 VIRAL CARD
        st.markdown("## 🔥 Viral Prediction")

        if positive_percent > 70:
            st.success("🚀 High viral potential")
        elif positive_percent > 50:
            st.info("👍 Good engagement")
        else:
            st.error("⚠️ Needs improvement")

        # 💬 TOP COMMENTS (CARDS)
        st.markdown("## 💬 Top Insights")

        col1, col2 = st.columns(2)

        with col1:
            if pos_comments:
                st.success(pos_comments[0])

        with col2:
            if neg_comments:
                st.error(neg_comments[0])

        # 💬 FULL BREAKDOWN
        st.markdown("## 💬 Comment Breakdown")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 😊 Positive")
            for c in pos_comments[:5]:
                st.success(c)

        with col2:
            st.markdown("### 😡 Negative")
            for c in neg_comments[:5]:
                st.error(c)

        # 🔥 KEYWORDS
        st.markdown("## 🔥 Audience Keywords")

        word_counts = Counter(all_words)
        common_words = word_counts.most_common(5)

        for word, count in common_words:
            st.write(f"🔹 {word}: {count}")

        # 📊 PIE CHART
        st.markdown("## 📊 Sentiment Distribution")

        fig, ax = plt.subplots()
        ax.pie([positive, negative], labels=["Positive", "Negative"], autopct='%1.1f%%')
        st.pyplot(fig)

        # 📄 DOWNLOAD
        st.markdown("## 📄 Export Report")

        report = f"""
Total Comments: {total}
Positive: {positive}
Negative: {negative}
Viral Score: {round(positive_percent,2)}%
Top Keywords: {common_words}
"""

        st.download_button("⬇️ Download Report", report, file_name="report.txt")

    else:
        st.warning("No comments found")
