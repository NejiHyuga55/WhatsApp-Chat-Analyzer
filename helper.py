from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji

extract = URLExtract()


# 1) Chat Statistics

def fetch_stats(selected_user, df):
    """Return message count, word count, media count, link count."""
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    num_messages = df.shape[0]

    # count words
    words = []
    for msg in df["message"]:
        words.extend(msg.split())

    # media messages
    media = df[df["message"] == "<Media omitted>\n"].shape[0]

    # links
    links = []
    for msg in df["message"]:
        links.extend(extract.find_urls(msg))

    return num_messages, len(words), media, len(links)


# 2) Most Busy Users

def most_busy_users(df):
    x = df["user"].value_counts().head()
    df_percent = (
        round((df["user"].value_counts() / df.shape[0]) * 100, 2)
        .reset_index()
        .rename(columns={"index": "name", "user": "percent"})
    )
    return x, df_percent


# 3) WordCloud

def create_wordcloud(selected_user, df):
    with open("stop_hinglish.txt", "r", encoding="utf-8") as f:
        stop_words = f.read()

    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    temp = df[(df["user"] != "group_notification") &
              (df["message"] != "<Media omitted>\n")]

    def clean(message):
        return " ".join(
            [w for w in message.lower().split() if w not in stop_words]
        )

    temp["message"] = temp["message"].apply(clean)

    wc = WordCloud(width=500, height=500,
                   background_color="white", min_font_size=10)

    return wc.generate(temp["message"].str.cat(sep=" "))


# 4) Most Common Words

def most_common_words(selected_user, df):
    with open("stop_hinglish.txt", "r", encoding="utf-8") as f:
        stop_words = f.read()

    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    temp = df[(df["user"] != "group_notification") &
              (df["message"] != "<Media omitted>\n")]

    words = []
    for msg in temp["message"]:
        for w in msg.lower().split():
            if w not in stop_words:
                words.append(w)

    return pd.DataFrame(Counter(words).most_common(20))


# 5) Emoji Analysis

def emoji_helper(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    emojis = []
    for msg in df["message"]:
        emojis.extend([c for c in msg if c in emoji.EMOJI_DATA])

    return pd.DataFrame(Counter(emojis).most_common(len(set(emojis))))


# 6) Timelines

def monthly_timeline(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    timeline = df.groupby(["year", "month_num", "month"]).count()["message"].reset_index()
    timeline["time"] = timeline["month"] + "-" + timeline["year"].astype(str)
    return timeline


def daily_timeline(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    return df.groupby("only_date").count()["message"].reset_index()


# 7) Activity Maps

def week_activity_map(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    return df["day_name"].value_counts()


def month_activity_map(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    return df["month"].value_counts()


def activity_heatmap(selected_user, df):
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    return df.pivot_table(
        index="day_name",
        columns="period",
        values="message",
        aggfunc="count"
    ).fillna(0)