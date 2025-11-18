import re
import pandas as pd


def preprocess(data):
    """
    Converts WhatsApp exported chat into a clean DataFrame.
    Extracts: date, user, message, and time features.
    """

    pattern = r"(\d{1,2}/\d{1,2}/\d{4},\s*\d{1,2}:\d{2}\s*[APap][Mm])\s-\s"

    dates = re.findall(pattern, data)
    messages = re.split(pattern, data)[1:]
    messages = messages[1::2]
    dates = dates[:len(messages)]

    df = pd.DataFrame({"user_message": messages, "message_date": dates})

    # --- convert date ---
    df["message_date"] = (
        df["message_date"]
        .astype(str)
        .str.replace("\u202f", " ", regex=True)
    )

    df["message_date"] = pd.to_datetime(
        df["message_date"],
        format="%d/%m/%Y, %I:%M %p",
        errors="coerce"
    )

    df.rename(columns={"message_date": "date"}, inplace=True)

    # --- separate user & message ---
    users = []
    msgs = []

    for msg in df["user_message"]:
        entry = re.split(r"([\w\W]+?):\s", msg)

        if len(entry) > 2:
            users.append(entry[1])
            msgs.append(entry[2])
        else:

            users.append("group_notification")
            msgs.append(entry[0])

    df["user"] = users
    df["message"] = msgs
    df.drop(columns=["user_message"], inplace=True)

    # --- remove system messages ---
    df = df[df["user"] != "group_notification"].reset_index(drop=True)

    # --- extract date parts ---
    df["only_date"] = df["date"].dt.date
    df["year"] = df["date"].dt.year
    df["month_num"] = df["date"].dt.month
    df["month"] = df["date"].dt.month_name()
    df["day"] = df["date"].dt.day
    df["day_name"] = df["date"].dt.day_name()
    df["hour"] = df["date"].dt.hour
    df["minute"] = df["date"].dt.minute

    # --- period for heatmap ---
    period = []
    for hour in df["hour"]:
        if hour == 23:
            period.append("23-00")
        elif hour == 0:
            period.append("00-01")
        else:
            period.append(f"{hour:02d}-{(hour+1)%24:02d}")

    df["period"] = period

    return df