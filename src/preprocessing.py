
import re
import html
import pandas as pd


def clean_text(text: str) -> str:
    """
    Normalize review text before TF-IDF feature extraction.
    Preserve meaningful words such as 'not' and 'never'.
    """
    if pd.isna(text):
        return ""

    text = str(text)
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Replace URLs with a common token
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " URL ",
        text,
        flags=re.IGNORECASE
    )

    # Normalize case and whitespace
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()

    return text
