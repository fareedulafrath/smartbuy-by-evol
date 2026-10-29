
import re
import html
import sqlite3
import unicodedata
from contextlib import closing
from pathlib import Path

import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "shopping_assistant.db"

ASPECT_KEYWORDS = {
    "Battery": [
        "battery", "battery life", "battery backup",
        "charging", "charge", "drain", "mah",
    ],
    "Camera": [
        "camera", "photo", "photos", "picture", "pictures",
        "selfie", "video recording", "night mode",
    ],
    "Performance": [
        "performance", "processor", "snapdragon", "dimensity",
        "lag", "lagging", "slow", "fast", "gaming",
        "heating", "overheating", "ram",
    ],
    "Display": [
        "display", "screen", "amoled", "oled", "brightness",
        "resolution", "refresh rate", "hz",
    ],
    "Build & Design": [
        "build quality", "design", "weight", "lightweight",
        "heavy", "durable", "plastic", "glass", "body",
    ],
    "Value for Money": [
        "value for money", "worth the price", "price",
        "budget", "affordable", "expensive", "overpriced",
        "cost",
    ],
    "Software & Updates": [
        "software", "update", "updates", "android",
        "bloatware", "ads", "interface", "ui",
    ],
    "Audio & Connectivity": [
        "speaker", "speakers", "sound", "audio", "volume",
        "network", "signal", "wifi", "wi-fi", "bluetooth",
        "5g", "call quality",
    ],
}

POSITIVE_PHRASES = [
    "good", "great", "excellent", "amazing", "awesome",
    "love", "loved", "best", "fantastic", "superb",
    "smooth", "fast", "clear", "bright", "durable",
    "impressive", "reliable", "satisfied", "worth it",
    "value for money", "long lasting", "long-lasting",
    "works well", "working well", "good quality",
    "battery backup is good", "battery life is good",
    "no issues", "no problem", "highly recommend",
    "easy to use", "quick charging",
]

NEGATIVE_PHRASES = [
    "bad", "poor", "worst", "terrible", "disappointing",
    "disappointed", "hate", "slow", "lag", "lagging",
    "overheating", "heating issue", "drains quickly",
    "battery drain", "poor battery", "low quality",
    "not working", "does not work", "doesn't work",
    "stopped working", "defective", "damaged", "broken",
    "waste of money", "not worth", "unhappy",
    "unreliable", "blurry", "blurry photos",
    "weak signal", "connectivity issue", "too expensive",
    "frequent issues", "not satisfied", "poor performance",
]

NEGATION_PATTERN = re.compile(
    r"\b(?:not|no|never|hardly|isn't|wasn't|doesn't|"
    r"don't|didn't|cannot|can't|without)\b",
    re.IGNORECASE,
)

SHORT_ASPECT_KEYWORDS = {"ram", "hz", "5g", "ui", "mah"}


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_review_text(text):
    """Clean review formatting while preserving readable meaning."""

    if not isinstance(text, str) or not text.strip():
        return ""

    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    text = re.sub(
        r"(?:={2,}|_{2,}|\*{2,}|-{4,}|>{1,2})",
        " ",
        text,
    )

    cleaned = []

    for character in text:
        category = unicodedata.category(character)

        # Remove emoji and non-text symbols.
        if category in {"So", "Cs", "Co"}:
            continue

        if category.startswith("C") and character not in "\n\t":
            continue

        cleaned.append(character)

    text = "".join(cleaned)

    text = re.sub(r"[ \t]*[|~^`]+[ \t]*", " ", text)
    text = re.sub(r"([!?.,])\1{2,}", r"\1", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,;:])(?=\S)", r"\1 ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)

    return text.strip(" \t\n-_ |")


def extract_review_sentences(text):
    """
    Split cleaned review text into sentence-sized units.

    Sentence splitting is heuristic and does not establish how many
    distinct customers contributed the text.
    """

    cleaned = clean_review_text(text)

    if not cleaned:
        return []

    cleaned = re.sub(
        r"\bReview\s+\d+\s*:",
        lambda match: "\n" + match.group(0),
        cleaned,
        flags=re.IGNORECASE,
    )

    parts = re.split(r"(?<=[.!?])\s+|\n+", cleaned)

    sentences = []

    for part in parts:
        sentence = part.strip(" \t\r\n,;:-")

        if len(sentence) >= 3:
            sentences.append(sentence)

    return sentences


def get_review_text(review_title, review_content):
    """Combine and clean a review title and its content."""

    title = clean_review_text(review_title)
    content = clean_review_text(review_content)

    if title and content:
        return f"{title}\n\n{content}"

    return title or content


# =========================================================
# DATABASE ACCESS
# =========================================================

def load_review_products():
    """Read product and review fields from the existing SQLite database."""

    if not DB_PATH.is_file():
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. "
            "Run setup_database.py first."
        )

    try:
        with closing(
            sqlite3.connect(
                DB_PATH.as_uri() + "?mode=ro",
                uri=True,
                timeout=10,
            )
        ) as connection:
            products = pd.read_sql_query(
                """
                SELECT
                    product_id,
                    product_name,
                    discounted_price,
                    rating,
                    review_title,
                    review_content,
                    img_link,
                    product_link
                FROM products
                """,
                connection,
            )

    except (sqlite3.Error, pd.errors.DatabaseError) as error:
        raise RuntimeError(
            "Unable to read product reviews from the database. "
            "Check the products table and its column names."
        ) from error

    if products.empty:
        return products

    products["product_id"] = (
        products["product_id"].astype("string").str.strip()
    )

    products["product_name"] = (
        products["product_name"].fillna("").astype(str).str.strip()
    )

    products["discounted_price"] = pd.to_numeric(
        products["discounted_price"],
        errors="coerce",
    )

    products["rating"] = pd.to_numeric(
        products["rating"],
        errors="coerce",
    )

    products = products.dropna(
        subset=["product_id", "discounted_price"]
    ).copy()

    products = products.loc[
        products["product_id"].ne("")
        & products["product_name"].ne("")
        & products["discounted_price"].ge(0)
    ].copy()

    for column in (
        "review_title",
        "review_content",
        "img_link",
        "product_link",
    ):
        if column not in products.columns:
            products[column] = ""
        products[column] = products[column].fillna("").astype(str)

    return products.drop_duplicates(
        subset=["product_id"],
        keep="first",
    ).reset_index(drop=True)


# =========================================================
# ASPECT EXTRACTION
# =========================================================

def find_aspects(sentence):
    """Return the product aspects mentioned in a sentence."""

    sentence_lower = str(sentence).casefold()
    found = []

    for aspect, keywords in ASPECT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in SHORT_ASPECT_KEYWORDS:
                pattern = r"\b" + re.escape(keyword) + r"\b"

                if re.search(pattern, sentence_lower):
                    found.append(aspect)
                    break

            elif keyword in sentence_lower:
                found.append(aspect)
                break

    return found


# =========================================================
# RULE-BASED SENTIMENT
# =========================================================

def phrase_matches(sentence, phrases):
    """Find sentiment phrases with simple word-boundary matching."""

    sentence_lower = str(sentence).casefold()
    matches = []

    for phrase in phrases:
        pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"

        if re.search(pattern, sentence_lower):
            matches.append(phrase)

    return matches


def classify_sentiment(sentence):
    """
    Classify a sentence as Positive, Negative, Mixed, or Neutral.

    This is a keyword-based heuristic, not a trained sentiment model.
    """

    positive = phrase_matches(sentence, POSITIVE_PHRASES)
    negative = phrase_matches(sentence, NEGATIVE_PHRASES)

    if positive and negative:
        return "Mixed"

    if not positive and not negative:
        return "Neutral"

    sentence_lower = str(sentence).casefold()

    # Detect common negated sentiment phrases, such as "not good".
    negation_pattern = re.compile(
        r"\b(?:not|never|isn't|wasn't|doesn't|don't|"
        r"didn't|can't|cannot|hardly)\b"
    )

    for phrase in positive:
        pattern = (
            r"\b(?:not|never|isn't|wasn't|doesn't|don't|"
            r"didn't|can't|cannot|hardly)"
            r"\s+(?:\w+\s+){0,2}"
            + re.escape(phrase)
            + r"\b"
        )

        if re.search(pattern, sentence_lower):
            return "Negative"

    for phrase in negative:
        pattern = (
            r"\b(?:not|never|isn't|wasn't|doesn't|don't|"
            r"didn't|can't|cannot|hardly)"
            r"\s+(?:\w+\s+){0,2}"
            + re.escape(phrase)
            + r"\b"
        )

        if re.search(pattern, sentence_lower):
            return "Positive"

    # Handle a negation near a sentiment phrase conservatively.
    if negation_pattern.search(sentence_lower):
        if positive:
            return "Negative"
        if negative:
            return "Positive"

    if positive:
        return "Positive"

    return "Negative"


# =========================================================
# REVIEW ANALYSIS
# =========================================================

def analyse_reviews(review_title, review_content):
    """
    Create structured review observations.

    Each observation contains:
      aspect, sentiment, sentence
    """

    combined = get_review_text(review_title, review_content)
    sentences = extract_review_sentences(combined)

    observations = []
    seen = set()

    for sentence in sentences:
        sentence_key = sentence.casefold()

        if sentence_key in seen:
            continue

        seen.add(sentence_key)

        aspects = find_aspects(sentence)
        sentiment = classify_sentiment(sentence)

        if not aspects:
            aspects = ["Overall Experience"]

        for aspect in aspects:
            observations.append(
                {
                    "aspect": aspect,
                    "sentiment": sentiment,
                    "sentence": sentence,
                }
            )

    return observations


def build_aspect_summary(observations):
    """Summarize positive, negative, mixed, and neutral evidence by aspect."""

    columns = [
        "Aspect",
        "Positive mentions",
        "Negative mentions",
        "Mixed mentions",
        "Neutral mentions",
        "Evidence sentences",
    ]

    if not observations:
        return pd.DataFrame(columns=columns)

    data = pd.DataFrame(observations)
    rows = []

    for aspect, group in data.groupby("aspect", sort=False):
        rows.append(
            {
                "Aspect": aspect,
                "Positive mentions": int(
                    (group["sentiment"] == "Positive").sum()
                ),
                "Negative mentions": int(
                    (group["sentiment"] == "Negative").sum()
                ),
                "Mixed mentions": int(
                    (group["sentiment"] == "Mixed").sum()
                ),
                "Neutral mentions": int(
                    (group["sentiment"] == "Neutral").sum()
                ),
                "Evidence sentences": int(len(group)),
            }
        )

    return pd.DataFrame(rows, columns=columns).sort_values(
        by=["Evidence sentences", "Aspect"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)


def detect_conflicts(observations, minimum_mentions=1):
    """
    Find aspects that contain both positive and negative evidence.

    This detects conflicting statements in the supplied text. It cannot
    confirm whether the statements came from different customers.
    """

    columns = [
        "Aspect",
        "Positive evidence",
        "Negative evidence",
        "Conflict status",
    ]

    if not observations:
        return pd.DataFrame(columns=columns)

    if (
        isinstance(minimum_mentions, bool)
        or not isinstance(minimum_mentions, int)
        or minimum_mentions < 1
    ):
        raise ValueError("minimum_mentions must be a positive integer.")

    data = pd.DataFrame(observations)
    conflicts = []

    for aspect, group in data.groupby("aspect", sort=False):
        positive = (
            group.loc[
                group["sentiment"] == "Positive", "sentence"
            ]
            .drop_duplicates()
            .tolist()
        )

        negative = (
            group.loc[
                group["sentiment"] == "Negative", "sentence"
            ]
            .drop_duplicates()
            .tolist()
        )

        if (
            len(positive) >= minimum_mentions
            and len(negative) >= minimum_mentions
        ):
            conflicts.append(
                {
                    "Aspect": aspect,
                    "Positive evidence": positive,
                    "Negative evidence": negative,
                    "Conflict status": "Conflicting opinions detected",
                }
            )

    return pd.DataFrame(conflicts, columns=columns)


def analyse_product_reviews(product):
    """
    Analyse one product and return the structure expected by preview_products.py.

    Returns:
      observations: list of aspect/sentiment/evidence dictionaries
      summary: DataFrame of sentiment counts by aspect
      conflicts: DataFrame of conflicting positive/negative evidence
    """

    observations = analyse_reviews(
        product.get("review_title", ""),
        product.get("review_content", ""),
    )

    return {
        "observations": observations,
        "summary": build_aspect_summary(observations),
        "conflicts": detect_conflicts(observations),
    }


# =========================================================
# LOCAL TEST
# =========================================================

def main():
    print("SmartBuy — Review Insights & Conflict Detection")
    print("-" * 52)

    sample = (
        "Good battery life!!! ===== Camera is okay 😊 "
        "Performance is not bad. Battery drains quickly. "
        "The camera is excellent."
    )

    print("\nCleaned sample:")
    print(clean_review_text(sample))

    observations = analyse_reviews("", sample)

    print("\nAspect sentiment observations:")

    if observations:
        for item in observations:
            print(
                f"- {item['aspect']} | {item['sentiment']} | "
                f"{item['sentence']}"
            )
    else:
        print("No review evidence found.")

    print("\nAspect summary:")
    summary = build_aspect_summary(observations)

    if summary.empty:
        print("No summary available.")
    else:
        print(summary.to_string(index=False))

    print("\nConflicting opinions:")
    conflicts = detect_conflicts(observations)

    if conflicts.empty:
        print("No conflicting positive and negative evidence detected.")
    else:
        for _, row in conflicts.iterrows():
            print(f"\n{row['Aspect']}:")

            print("Positive evidence:")
            for sentence in row["Positive evidence"]:
                print(f"  + {sentence}")

            print("Negative evidence:")
            for sentence in row["Negative evidence"]:
                print(f"  - {sentence}")


if __name__ == "__main__":
    main()
