import os
import re
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import preprocessor
import helper

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Segoe UI Emoji"

# Load Emotion Model

MODEL_PATH = "trained_models/emotion_pipeline_svm.joblib"
ENCODER_PATH = "trained_models/label_encoder_svm.joblib"


@st.cache_resource
def load_emotion_model():
    try:
        pipe = joblib.load(MODEL_PATH)
        enc = joblib.load(ENCODER_PATH)
        return pipe, enc
    except:
        return None, None

# Text cleaning for ML model

def clean_for_emotion(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^0-9A-Za-z\u0900-\u097F\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def predict_emotions_for_df(pipeline, encoder, df):
    out = df.copy()
    out["clean_message"] = out["message"].apply(clean_for_emotion)

    nonempty = out["clean_message"].str.len() > 0

    if pipeline is None:
        out["emotion"] = "neutral"
        return out

    preds = pipeline.predict(out.loc[nonempty, "clean_message"])
    out.loc[nonempty, "emotion"] = encoder.inverse_transform(preds)
    out.loc[~nonempty, "emotion"] = "neutral"

    # Force: Only these 3 classes allowed
    out["emotion"] = out["emotion"].replace({
        "pos": "positive",
        "neg": "negative",
        "neu": "neutral"
    })

    return out

# Streamlit UI

st.set_page_config(page_title="WhatsApp Analyzer", layout="wide")
st.sidebar.title("📁 Menu")

tool = st.sidebar.radio("Choose Tool:", ["WhatsApp Chat Analyzer",
                                         "WhatsApp Emotion Analyzer"])

uploaded_file = st.sidebar.file_uploader("📤 Upload WhatsApp Chat (.txt)",
                                         type=["txt"])

st.title("💬 WhatsApp Chat + Emotion Analyzer")

if uploaded_file is None:
    st.info("Upload a .txt chat export to continue.")
    st.stop()


# Preprocess WhatsApp Chat

raw = uploaded_file.getvalue().decode("utf-8", errors="ignore")
df = preprocessor.preprocess(raw)

# user list
users = df["user"].unique().tolist()
if "group_notification" in users:
    users.remove("group_notification")
users.sort()
users.insert(0, "Overall")

selected_user = st.sidebar.selectbox("Analyze for:", users)

pipeline, encoder = load_emotion_model()

# 1) CHAT ANALYZER

if tool == "WhatsApp Chat Analyzer":

    st.header("📱 WhatsApp Chat Analyzer")

    if st.sidebar.button("Show Chat Analysis"):

        df_chat = df.copy()
        if selected_user != "Overall":
            df_chat = df_chat[df_chat["user"] == selected_user]

        # ------- Stats -------
        st.subheader("📊 Top Statistics")
        num_messages, words, media, links = helper.fetch_stats(
            selected_user if selected_user != "Overall" else "Overall", df_chat
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Messages", num_messages)
        c2.metric("Words", words)
        c3.metric("Media Shared", media)
        c4.metric("Links Shared", links)

        # ------- Preview -------
        st.subheader("📄 Chat Preview")
        st.dataframe(df_chat[["date", "user", "message"]].head(50))

        # ------- Monthly Timeline -------
        st.subheader("📆 Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df_chat)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(timeline["time"], timeline["message"])
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # ------- Daily Timeline -------
        st.subheader("📅 Daily Timeline")
        daily = helper.daily_timeline(selected_user, df_chat)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(daily["only_date"], daily["message"])
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # ------- Activity Maps -------
        st.subheader("🕒 Activity Map")
        st.subheader("📅 Most Busy Day")

        col1, col2 = st.columns(2)

        with col1:
            busy_day = helper.week_activity_map(selected_user, df_chat)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values)
            plt.xticks(rotation=45)
            st.pyplot(fig)


        with col2:

            st.subheader("📆 Most Busy Month")

            busy_month = helper.month_activity_map(selected_user, df_chat)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values)
            plt.xticks(rotation=45)
            st.pyplot(fig)

        # ------- Heatmap -------
        st.subheader("🔥 Weekly Activity Heatmap")
        heat = helper.activity_heatmap(selected_user, df_chat)
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.heatmap(heat, cmap="Blues", ax=ax)
        st.pyplot(fig)

        # ------- Most Active Users -------
        st.subheader("🔥 Most Active Users")

        active = df_chat["user"].value_counts().head(10)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(active.index, active.values, color="crimson")
        ax.set_ylabel("Messages Sent")
        ax.set_title("Top 10 Most Active Users")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

        st.dataframe(active.reset_index().rename(columns={"index": "User", "user": "Messages"}))

        # ------- Wordcloud -------
        st.subheader("☁ Wordcloud")
        wc = helper.create_wordcloud(selected_user, df_chat)
        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)

        # ------- Most Common Words -------
        st.subheader("🔤 Most Common Words")
        mc = helper.most_common_words(selected_user, df_chat)
        fig, ax = plt.subplots()
        ax.barh(mc[0], mc[1])
        st.pyplot(fig)

        # ------- Emoji Analysis -------
        st.subheader("😊 Emoji Analysis")
        emoji_df = helper.emoji_helper(selected_user, df_chat)

        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(emoji_df)

        with c2:
            if not emoji_df.empty:
                top = emoji_df.head(10)
                fig, ax = plt.subplots()
                ax.pie(top[1], labels=[str(x) for x in top[0]],
                       autopct="%1.1f%%")
                st.pyplot(fig)
            else:
                st.info("No emojis found.")

# 2) EMOTION ANALYZER (3-CLASS)

else:
    st.header("🧠 WhatsApp Emotion Analyzer")

    if pipeline is None:
        st.error("Model missing! Run emotion_model_train.py first.")
        st.stop()

    df_pred = predict_emotions_for_df(pipeline, encoder, df)

    if selected_user != "Overall":
        df_show = df_pred[df_pred["user"] == selected_user]
    else:
        df_show = df_pred

    # ------- Summary -------
    st.subheader("📊 Emotion Summary")

    counts = df_show["emotion"].value_counts()[["positive", "neutral", "negative"]]

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📊 bar chart")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=counts.index, y=counts.values, palette="viridis")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with col2:
        st.subheader("📊 pie chart")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(counts.values,
               labels=counts.index,
               autopct="%1.1f%%",
               startangle=120,
               pctdistance=0.75,
               labeldistance=1.15)
        ax.axis("equal")
        st.pyplot(fig)

    # ------- User-wise -------
    st.subheader("👥 User-wise Emotion Distribution")

    # 🔥 NEW INSIGHTS: Most Positive, Negative & Active Users


    st.subheader("🏆 Most Positive / Negative / Active Users")

    # 1) Count emotions per user
    emotion_counts = df_pred.groupby(["user", "emotion"]).size().unstack(fill_value=0)

    # 2) User activity (total messages)
    activity_counts = df_pred["user"].value_counts()

    # ----- Most Positive Users -----
    st.markdown("### 😊 Most Positive Users")
    pos_sorted = emotion_counts["positive"].sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=pos_sorted.values, y=pos_sorted.index, palette="Greens")
    ax.set_xlabel("Positive Messages")
    ax.set_ylabel("User")
    st.pyplot(fig)

    # ----- Most Negative Users -----
    st.markdown("### 😡 Most Negative Users")
    neg_sorted = emotion_counts["negative"].sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=neg_sorted.values, y=neg_sorted.index, palette="Reds")
    ax.set_xlabel("Negative Messages")
    ax.set_ylabel("User")
    st.pyplot(fig)

    # ----- Most Active Users -----
    st.markdown("### 📣 Most Active Users")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=activity_counts.values, y=activity_counts.index, palette="Blues")
    ax.set_xlabel("Total Messages")
    ax.set_ylabel("User")
    st.pyplot(fig)

    table = df_pred.groupby(["user", "emotion"]).size().unstack(fill_value=0)
    st.dataframe(table)

    # 📌 Sentiment Score Per User (Research-Paper Style Graph)


    st.subheader("📈 Sentiment Score of Users (Positive vs Negative Balance)")

    # Counts of emotions per user
    emotion_counts = df_pred.groupby(["user", "emotion"]).size().unstack(fill_value=0)

    # Calculate sentiment score
    emotion_counts["sentiment_score"] = (
            (emotion_counts.get("positive", 0) - emotion_counts.get("negative", 0)) /
            (emotion_counts.sum(axis=1) + 1e-9)
    )

    # Sort by sentiment score
    sentiment_sorted = emotion_counts["sentiment_score"].sort_values(ascending=False)

    # Plot graph
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#4CAF50" if x > 0 else "#F44336" for x in sentiment_sorted]  # green for +ve, red for -ve

    ax.bar(sentiment_sorted.index, sentiment_sorted.values, color=colors)

    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Sentiment Score")
    ax.set_xlabel("User")
    ax.set_title("Sentiment Analysis of Users in WhatsApp Chat")

    plt.xticks(rotation=90)
    plt.tight_layout()
    st.pyplot(fig)

    # ------- Heatmap -------
    st.subheader("📆 Weekly Activity Heatmap (Predictions)")
    pivot = df_pred.pivot_table(index="day_name",
                                columns="period",
                                values="emotion",
                                aggfunc="count").fillna(0)
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(pivot, cmap="Blues", ax=ax)
    st.pyplot(fig)

    # ------- Download -------
    st.subheader("⬇ Download Predictions CSV")
    out = df_pred[["date", "user", "message", "emotion"]]
    st.download_button("Download emotion_predictions.csv",
                       out.to_csv(index=False),
                       "emotion_predictions.csv",
                       "text/csv")

    if st.checkbox("Show sample predicted messages"):
        st.dataframe(df_show.head(300))