import streamlit as st
from collections import Counter
import matplotlib.pyplot as plt
from googleapiclient.discovery import build
import re
from transformers import pipeline

# 🔥 Load AI model (only loads once)
sentiment_model = pipeline("sentiment-analysis")

# 🔹 Function to fetch YouTube comments
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


# 🚀 UI
st.title("🚀 Creator Comment Analyzer (AI + YouTube)")

video_url = st.text_input("📺 Paste YouTube Video Link:")
api_key = st.secrets["API_KEY"]

comments_input = st.text_area("✍️ Or paste comments manually (one per line):")

max_comments = st.slider("Number of comments to analyze", 50, 500, 200)

if st.button("Analyze Comments"):

    # 🔹 Choose source
    if video_url and api_key:
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
            # 🔥 AI SENTIMENT
            result = sentiment_model(comment[:512])[0]
            label = result['label']

            if label == "POSITIVE":
                positive += 1
            elif label == "NEGATIVE":
                negative += 1
            else:
                neutral += 1

            # 🔹 Clean words
            words = re.findall(r'\b[a-zA-Z]{4,}\b', comment.lower())
            all_words.extend(words)

        total = len(comments)

        # 📊 Results
        st.subheader("📊 Results")
        st.write(f"Total Comments: {total}")
        st.write(f"Positive: {positive}")
        st.write(f"Negative: {negative}")
        st.write(f"Neutral: {neutral}")

        # 📊 Pie Chart
        st.subheader("📊 Sentiment Distribution")

        labels = ['Positive', 'Negative', 'Neutral']
        sizes = [positive, negative, neutral]

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%')
        ax1.axis('equal')

        st.pyplot(fig1)

        # 🔥 Keywords
        st.subheader("🔥 Top Keywords")
        word_counts = Counter(all_words)
        common_words = word_counts.most_common(5)

        for word, count in common_words:
            st.write(f"{word} : {count}")

        # 📊 Bar Chart
        st.subheader("📊 Keyword Frequency")

        words = [w for w, c in common_words]
        counts = [c for w, c in common_words]

        fig2, ax2 = plt.subplots()
        ax2.bar(words, counts)
        st.pyplot(fig2)

        # 💡 Insight
        st.subheader("💡 Insight")
        if positive > negative:
            st.success("Audience is liking your content 👍")
        elif negative > positive:
            st.error("Audience is not happy 😬 Improve content")
        else:
            st.info("Mixed response 😐 Try experimenting")

        # 🚀 Content Suggestions
        st.subheader("🚀 Content Suggestions")

        top_words = [word for word, count in common_words]

        if top_words:
            st.write(f"👉 People are talking about: {', '.join(top_words)}")

        if any("edit" in word for word in top_words):
            st.write("👉 Improve or maintain your editing style.")

        if any("audio" in word for word in top_words):
            st.write("👉 Improve audio quality.")

        if any("content" in word for word in top_words):
            st.write("👉 Maintain content consistency.")

        if positive > negative:
            st.success("👉 Continue this type of content 👍")
        elif negative > positive:
            st.error("👉 Improve your content quality")
        else:
            st.info("👉 Try experimenting with new content ideas")

    else:
        st.warning("No comments found")
