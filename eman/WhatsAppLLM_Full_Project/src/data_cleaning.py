import re
import pandas as pd

CHAT_FILE = "data/chat.txt"

messages = []

pattern = r"(.+?), (.+?) - (.*?): (.*)"

with open(
    CHAT_FILE,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        line = line.strip()

        match = re.match(
            pattern,
            line
        )

        if not match:
            continue

        date = match.group(1)
        time = match.group(2)
        sender = match.group(3)
        message = match.group(4)

        # Convert to lowercase
        message = message.lower()

        # Remove extra spaces
        message = re.sub(
            r"\s+",
            " ",
            message
        )

        # Remove media/system messages
        if "media omitted" in message:
            continue

        if "image omitted" in message:
            continue

        if "video omitted" in message:
            continue

        if "audio omitted" in message:
            continue

        if "document omitted" in message:
            continue

        if "sticker omitted" in message:
            continue

        # Remove deleted messages
        if "you deleted this message" in message:
            continue

        if "this message was deleted" in message:
            continue

        # Remove URLs
        message = re.sub(
            r"http\S+",
            "",
            message
        )

        # Remove emails
        message = re.sub(
            r"\S+@\S+",
            "",
            message
        )

        # Remove tags
        message = re.sub(
            r"@\w+",
            "",
            message
        )

        message = message.strip()

        if len(message) == 0:
            continue

        messages.append({
            "date": date,
            "time": time,
            "sender": sender,
            "message": message
        })

df = pd.DataFrame(messages)

print(df.head())

print(
    "\nRows:",
    len(df)
)

df.to_csv(
    "data/cleaned_chat.csv",
    index=False
)

print(
    "\nSaved: data/cleaned_chat.csv"
)