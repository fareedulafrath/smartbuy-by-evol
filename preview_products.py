import base64
import html
import math
import sqlite3
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import streamlit as st

from recommendation_engine import (
    get_recommendations,
    explain_recommendation,
    validate_priorities,
    get_priority_label,
    calculate_value_scores,
    explain_value_score,
)
from review_insights import analyse_product_reviews, get_review_text
from smartbuy_agent import ask_smartbuy


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "shopping_assistant.db"
LOGO_PATH = BASE_DIR / "evol_logo.png"
BANNER_PATH = BASE_DIR / "smartbuy_banner.png"

st.set_page_config(
    page_title="SmartBuy | EVOL",
    page_icon=str(LOGO_PATH) if LOGO_PATH.is_file() else "🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURE_PAGES = [
    "Browse Smartphones",
    "Smart Purchase Decision",
    "Review Insights & Conflict Detection",
    "Budget Alternatives & Best Value",
    "Purchase Regret Prevention",
]

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        min-width: 260px;
    }

    .product-title {
        min-height: 3.4em;
        height: 3.4em;
        line-height: 1.7em;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow-wrap: anywhere;
        font-size: .98rem;
        font-weight: 600;
    }

    .product-image-slot {
        height: 190px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #777;
        font-size: .9rem;
    }

    [data-testid="stImage"] {
        display: flex;
        justify-content: center;
        align-items: center;
    }

    [data-testid="stImage"] img {
        object-fit: contain;
        object-position: center;
        max-height: 200px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px;
    }

    .smartbuy-hero {
        width: 100%;
        min-height: 210px;
        border-radius: 16px;
        overflow: hidden;
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        display: flex;
        align-items: center;
        padding: 28px 34px;
        box-sizing: border-box;
        margin-bottom: 24px;
    }

    .smartbuy-hero-title {
        color: #101828;
        font-size: clamp(2rem, 4vw, 3.2rem);
        font-weight: 800;
        line-height: 1.05;
        margin: 0;
    }

    .smartbuy-hero-subtitle {
        color: #344054;
        font-size: 1rem;
        margin-top: 12px;
        max-width: 390px;
    }

    .smartbuy-hero-brand {
        color: #344054;
        font-weight: 600;
        margin-top: 6px;
        font-size: 1.1rem;
    }

    @media (max-width: 700px) {
        .smartbuy-hero {
            min-height: 160px;
            padding: 20px;
            background-position: 60% center;
        }

        .smartbuy-hero-subtitle {
            max-width: 220px;
            font-size: .85rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# DATABASE
# =========================================================

@st.cache_data
def load_products(database_path):
    with sqlite3.connect(database_path, timeout=10) as connection:
        data = pd.read_sql_query("SELECT * FROM products", connection)

    required = {"product_id", "product_name", "discounted_price"}
    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            "The product database is missing required information."
        )

    for col in [
        "discounted_price",
        "actual_price",
        "rating",
        "rating_count",
        "discount_percentage",
    ]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data["product_id"] = data["product_id"].astype("string").str.strip()
    data["product_name"] = data["product_name"].astype("string").str.strip()

    valid_price = (
        data["discounted_price"].notna()
        & data["discounted_price"].ge(0)
        & data["discounted_price"].map(
            lambda x: math.isfinite(float(x))
        )
    )

    valid = (
        data["product_id"].notna()
        & data["product_id"].ne("")
        & data["product_name"].notna()
        & data["product_name"].ne("")
        & valid_price
    )

    data = data.loc[valid].copy()

    defaults = {
        "actual_price": float("nan"),
        "rating": float("nan"),
        "rating_count": float("nan"),
        "discount_percentage": float("nan"),
        "about_product": "",
        "review_title": "",
        "review_content": "",
        "img_link": "",
        "product_link": "",
    }

    for col, default in defaults.items():
        if col not in data.columns:
            data[col] = default

    valid_rating = data["rating"].notna() & data["rating"].between(0, 5)
    data.loc[~valid_rating, "rating"] = float("nan")

    for col in [
        "about_product",
        "review_title",
        "review_content",
        "img_link",
        "product_link",
    ]:
        data[col] = data[col].fillna("").astype(str).str.strip()

    return data.drop_duplicates(
        subset=["product_id"], keep="first"
    ).reset_index(drop=True)


try:
    if not DB_PATH.is_file():
        raise FileNotFoundError

    df = load_products(str(DB_PATH))

except FileNotFoundError:
    st.error(
        "The product database could not be found. "
        "Please check that shopping_assistant.db is in the project folder."
    )
    st.stop()

except (
    sqlite3.Error,
    pd.errors.DatabaseError,
    ValueError,
    KeyError,
) as error:
    st.error(f"Unable to load smartphone data: {error}")
    st.stop()

if df.empty:
    st.warning("No smartphones are currently available.")
    st.stop()


# =========================================================
# SESSION STATE
# =========================================================

SESSION_DEFAULTS = {
    "selected_product": None,
    "smartbuy_decision": None,
    "smartbuy_page": "Browse Smartphones",
    "main_navigation": "Browse Smartphones",
    "combined_comparison_result": None,
    "smartbuy_chat_history": [],
}

for key, value in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

for old_page in ("Budget-Friendly Alternatives", "Best Value for Money"):
    for key in ("smartbuy_page", "main_navigation"):
        if st.session_state[key] == old_page:
            st.session_state[key] = "Budget Alternatives & Best Value"


# =========================================================
# COMMON HELPERS
# =========================================================

def safe_text(value, fallback=""):
    if value is None:
        return fallback

    try:
        if pd.isna(value):
            return fallback
    except (TypeError, ValueError):
        pass

    value = str(value).strip()
    return fallback if value.lower() in {"nan", "none"} else value


def valid_image_url(value):
    try:
        parsed = urlparse(safe_text(value))
        return parsed.scheme in {"https", "http"} and bool(parsed.netloc)
    except ValueError:
        return False


def valid_product_url(value):
    try:
        parsed = urlparse(safe_text(value))
        return parsed.scheme == "https" and bool(parsed.netloc)
    except ValueError:
        return False


def format_price(value):
    try:
        number = float(value)
        return (
            f"₹{number:,.0f}"
            if math.isfinite(number)
            else "Price unavailable"
        )
    except (TypeError, ValueError):
        return "Price unavailable"


def show_product_image(product, width=180, slot_height=190):
    url = product.get("img_link", "")

    if valid_image_url(url):
        try:
            st.image(url, width=width)
            return
        except Exception:
            pass

    st.markdown(
        f'<div class="product-image-slot" style="height:{slot_height}px">'
        "Image unavailable</div>",
        unsafe_allow_html=True,
    )


def show_price(product):
    price_value = product.get("discounted_price")
    st.markdown(f"### {format_price(price_value)}")

    try:
        price = float(price_value)
        actual = float(product.get("actual_price"))

        if math.isfinite(actual) and actual > price > 0:
            discount = round((actual - price) / actual * 100)
            st.caption(f"₹{actual:,.0f} · {discount}% off")

    except (TypeError, ValueError):
        pass


def show_rating(product):
    rating = product.get("rating")

    if pd.notna(rating):
        st.write(f"⭐ {float(rating):.1f} / 5")
    else:
        st.caption("Rating unavailable")


def show_product_link(product, label="View Product"):
    link = product.get("product_link", "")

    if valid_product_url(link):
        st.link_button(label, link, width="stretch")


def open_product_details(product_id):
    st.session_state["selected_product"] = str(product_id)
    st.session_state["smartbuy_page"] = "Browse Smartphones"
    st.session_state["main_navigation"] = "Browse Smartphones"


def back_to_browsing():
    st.session_state["selected_product"] = None
    st.session_state["smartbuy_page"] = "Browse Smartphones"
    st.session_state["main_navigation"] = "Browse Smartphones"


def clear_smartbuy_decision():
    st.session_state["smartbuy_decision"] = None


def show_product_details(product):
    st.header(safe_text(product.get("product_name"), "Smartphone"))

    image_col, detail_col = st.columns([1, 2], gap="large")

    with image_col:
        show_product_image(product, width=300, slot_height=300)
        show_price(product)
        show_rating(product)
        show_product_link(product)

        st.button(
            "← Back to products",
            width="stretch",
            key="back_to_products",
            on_click=back_to_browsing,
        )

    with detail_col:
        st.subheader("Specifications")
        description = safe_text(product.get("about_product"))
        st.write(description or "Specifications unavailable.")

        with st.expander("Customer reviews"):
            title = safe_text(product.get("review_title"))
            if title:
                st.markdown(f"**{title}**")
            st.write(
                safe_text(product.get("review_content"))
                or "No review text available."
            )


# =========================================================
# HEADER BANNER
# =========================================================

if BANNER_PATH.is_file():
    try:
        with open(BANNER_PATH, "rb") as banner_file:
            banner_base64 = base64.b64encode(
                banner_file.read()
            ).decode("utf-8")

        st.markdown(
            f"""
            <div class="smartbuy-hero"
                 style="background-image:url(
                 'data:image/png;base64,{banner_base64}');">
                <div>
                    <div class="smartbuy-hero-title">SmartBuy</div>
                    <div class="smartbuy-hero-brand">by EVOL</div>
                    <div class="smartbuy-hero-subtitle">
                        Find the right smartphone for your needs and budget.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    except OSError:
        st.title("SmartBuy")
        st.caption("Find the right smartphone for your needs and budget.")
else:
    st.title("SmartBuy")
    st.caption("Find the right smartphone for your needs and budget.")


# =========================================================
# PRICE-BAND HELPERS
# =========================================================

def get_preferred_price_floor(budget):
    budget = float(budget)

    if budget <= 15000:
        floor = budget - 10000
    elif budget <= 20000:
        floor = budget - 5000
    elif budget <= 25000:
        floor = 15000
    elif budget <= 35000:
        floor = 20000
    else:
        floor = budget - 12000

    return max(0.0, min(floor, budget))


def filter_products_for_budget_band(
    products,
    budget,
    min_rating=0.0,
    minimum_products=3,
):
    budget = float(budget)
    min_rating = float(min_rating)

    eligible = products[
        products["discounted_price"].notna()
        & products["discounted_price"].gt(0)
        & products["discounted_price"].le(budget)
    ].copy()

    if min_rating > 0:
        eligible = eligible[
            eligible["rating"].notna()
            & eligible["rating"].ge(min_rating)
        ].copy()

    floor = get_preferred_price_floor(budget)

    if eligible.empty:
        return eligible, floor, False

    preferred = eligible[
        eligible["discounted_price"].ge(floor)
    ].copy()

    if len(preferred) >= minimum_products:
        return preferred, floor, False

    for fallback in [
        max(0.0, floor - 5000),
        max(0.0, floor - 10000),
        0.0,
    ]:
        expanded = eligible[
            eligible["discounted_price"].ge(fallback)
        ].copy()

        if len(expanded) >= minimum_products:
            return expanded, fallback, True

    return eligible, 0.0, True


def add_display_value_score(products):
    result = products.copy()

    base = pd.to_numeric(
        result.get(
            "value_score",
            pd.Series(float("nan"), index=result.index),
        ),
        errors="coerce",
    ).clip(0, 100)

    rating = pd.to_numeric(
        result.get(
            "rating",
            pd.Series(float("nan"), index=result.index),
        ),
        errors="coerce",
    )
    rating_score = (rating / 5 * 100).clip(0, 100)

    coverage = pd.to_numeric(
        result.get(
            "specification_coverage",
            pd.Series(float("nan"), index=result.index),
        ),
        errors="coerce",
    ).clip(0, 100)

    weighted_sum = base.fillna(0) * 0.50
    weight_sum = base.notna().astype(float) * 0.50

    weighted_sum += rating_score.fillna(0) * 0.35
    weight_sum += rating_score.notna().astype(float) * 0.35

    weighted_sum += coverage.fillna(0) * 0.15
    weight_sum += coverage.notna().astype(float) * 0.15

    result["display_value_score"] = (
        weighted_sum / weight_sum.replace(0, float("nan"))
    ).clip(0, 100)

    return result

# =========================================================
# PRODUCT CARDS
# =========================================================

def render_product_grid(product_data, key_prefix="product"):
    records = product_data.to_dict("records")

    for start in range(0, len(records), 3):
        columns = st.columns(3, gap="medium")

        for column, product in zip(columns, records[start:start + 3]):
            with column:
                with st.container(border=True):
                    show_product_image(product)

                    name = html.escape(
                        safe_text(product.get("product_name"), "Smartphone")
                    )
                    st.markdown(
                        f'<div class="product-title">{name}</div>',
                        unsafe_allow_html=True,
                    )

                    show_price(product)
                    show_rating(product)
                    show_product_link(product)

                    st.button(
                        "View details",
                        key=f"{key_prefix}_{product['product_id']}",
                        width="stretch",
                        on_click=open_product_details,
                        args=(str(product["product_id"]),),
                    )


def render_recommendation_cards(recommendations, priority):
    if recommendations is None or recommendations.empty:
        st.info("No matching smartphones found.")
        return

    for _, product in recommendations.iterrows():
        with st.container(border=True):
            image_col, detail_col = st.columns([1, 3], gap="large")

            with image_col:
                show_product_image(product, width=190, slot_height=200)

            with detail_col:
                name = html.escape(
                    safe_text(product.get("product_name"), "Smartphone")
                )
                st.markdown(f"**{name}**")
                show_price(product)
                show_rating(product)

                score = product.get("recommendation_score")
                if pd.notna(score):
                    st.caption("Recommendation match")
                    st.write(f"{float(score):.1f}/100")

                try:
                    explanation = explain_recommendation(product, priority)
                    if explanation:
                        with st.expander("Why this phone?"):
                            st.write(explanation)
                except (
                    KeyError,
                    ValueError,
                    TypeError,
                    AttributeError,
                ):
                    pass

                show_product_link(product)

                st.button(
                    "View details",
                    key=f"recommendation_details_{product['product_id']}",
                    width="stretch",
                    on_click=open_product_details,
                    args=(str(product["product_id"]),),
                )


def render_comparison_cards(products, budget):
    records = products.to_dict("records")

    for start in range(0, len(records), 3):
        columns = st.columns(3, gap="medium")

        for column, product in zip(columns, records[start:start + 3]):
            with column:
                with st.container(border=True):
                    show_product_image(product)

                    name = html.escape(
                        safe_text(product.get("product_name"), "Smartphone")
                    )
                    st.markdown(
                        f'<div class="product-title">{name}</div>',
                        unsafe_allow_html=True,
                    )

                    show_price(product)
                    show_rating(product)

                    price = float(product["discounted_price"])
                    st.caption(
                        f"{format_price(max(0, budget - price))} "
                        "left in your budget"
                    )

                    score = product.get(
                        "display_value_score",
                        product.get("value_score"),
                    )
                    if pd.notna(score):
                        st.metric(
                            "Estimated value score",
                            f"{float(score):.1f}/100",
                        )

                    try:
                        explanation = explain_value_score(product)
                        if explanation:
                            with st.expander("Why this phone?"):
                                st.write(explanation)
                    except (
                        KeyError,
                        ValueError,
                        TypeError,
                        AttributeError,
                    ):
                        pass

                    show_product_link(product)

                    st.button(
                        "View details",
                        key=f"comparison_details_{product['product_id']}",
                        width="stretch",
                        on_click=open_product_details,
                        args=(str(product["product_id"]),),
                    )


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

current_page = st.session_state["smartbuy_page"]

if current_page not in FEATURE_PAGES:
    current_page = FEATURE_PAGES[0]
    st.session_state["smartbuy_page"] = current_page
    st.session_state["main_navigation"] = current_page

if LOGO_PATH.is_file():
    st.sidebar.image(str(LOGO_PATH), width=180)

st.sidebar.title("SmartBuy")
st.sidebar.caption("Your AI shopping assistant")
st.sidebar.subheader("Explore features")

selected_page = st.sidebar.radio(
    "Navigation",
    options=FEATURE_PAGES,
    index=FEATURE_PAGES.index(current_page),
    key="main_navigation",
    label_visibility="collapsed",
)

if selected_page != st.session_state["smartbuy_page"]:
    st.session_state["smartbuy_page"] = selected_page
    st.session_state["selected_product"] = None
    st.rerun()

st.sidebar.subheader("Shopping filters")

min_price = int(df["discounted_price"].min())
max_price = int(df["discounted_price"].max())

if min_price < max_price:
    browser_budget = st.sidebar.slider(
        "Maximum budget (₹)",
        min_value=min_price,
        max_value=max_price,
        value=max_price,
        step=500,
        key="browser_budget",
    )
else:
    browser_budget = max_price
    st.sidebar.caption(f"Available price: ₹{max_price:,}")

browser_rating = st.sidebar.selectbox(
    "Minimum rating",
    options=[0.0, 3.0, 3.5, 4.0, 4.5],
    index=0,
    format_func=lambda x: "Any rating" if x == 0 else f"{x}+ stars",
    key="browser_rating",
)

st.sidebar.caption("These filters apply to Browse Smartphones.")
st.sidebar.caption("EVOL · SmartBuy")


# =========================================================
# SMARTBUY AI CHAT
# =========================================================

with st.popover("💬 Chat with SmartBuy AI", use_container_width=True):
    st.subheader("SmartBuy AI Assistant")
    st.caption(
        "Ask about smartphones, purchase priorities, reviews, "
        "budget alternatives, and value for money."
    )

    if st.button(
        "Clear conversation",
        key="clear_smartbuy_chat",
        width="stretch",
    ):
        st.session_state["smartbuy_chat_history"] = []
        st.rerun()

    chat_history = st.session_state["smartbuy_chat_history"]

    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    with st.form("smartbuy_chat_form", clear_on_submit=True):
        chat_prompt = st.text_input(
            "Your message",
            placeholder="e.g. Suggest a good phone under ₹30,000",
            key="smartbuy_chat_prompt",
        )
        chat_submitted = st.form_submit_button(
            "Send",
            type="primary",
            width="stretch",
        )

    if chat_submitted:
        prompt = chat_prompt.strip()

        if not prompt:
            st.warning("Please enter a message.")
        else:
            st.session_state["smartbuy_chat_history"].append(
                {"role": "user", "content": prompt}
            )

            history_for_agent = st.session_state[
                "smartbuy_chat_history"
            ][:-1]

            try:
                with st.spinner("SmartBuy AI is thinking..."):
                    answer = ask_smartbuy(
                        prompt,
                        history=history_for_agent,
                    )

                if not isinstance(answer, str):
                    answer = str(answer)

                st.session_state["smartbuy_chat_history"].append(
                    {"role": "assistant", "content": answer}
                )

            except Exception:
                st.session_state["smartbuy_chat_history"].append(
                    {
                        "role": "assistant",
                        "content": (
                            "I'm unable to respond right now. "
                            "Please try again in a moment."
                        ),
                    }
                )

            st.rerun()


# =========================================================
# PAGE 1: BROWSE SMARTPHONES
# =========================================================

if selected_page == "Browse Smartphones":
    selected_id = st.session_state.get("selected_product")

    if selected_id is not None:
        selected = df[
            df["product_id"].astype(str) == str(selected_id)
        ]

        if not selected.empty:
            show_product_details(selected.iloc[0])
        else:
            st.warning("This smartphone could not be found.")
            st.button("Return to browsing", on_click=back_to_browsing)

        st.stop()

    st.header("Browse Smartphones")

    search_text = st.text_input(
        "Search smartphones",
        placeholder="Search by brand, model, or feature...",
        key="browser_search",
    )

    price_col, rating_col = st.columns(2)

    with price_col:
        price_sort = st.selectbox(
            "Sort by price",
            ["Recommended order", "Low to High", "High to Low"],
            key="price_sort",
        )

    with rating_col:
        rating_sort = st.selectbox(
            "Sort by rating",
            ["None", "High to Low", "Low to High"],
            key="rating_sort",
        )

    filtered_df = df[
        df["discounted_price"].le(browser_budget)
    ].copy()

    if browser_rating > 0:
        filtered_df = filtered_df[
            filtered_df["rating"].notna()
            & filtered_df["rating"].ge(browser_rating)
        ].copy()

    if search_text.strip():
        search_values = (
            filtered_df["product_name"].fillna("").astype(str)
            + " "
            + filtered_df["about_product"].fillna("").astype(str)
        )

        filtered_df = filtered_df[
            search_values.str.contains(
                search_text.strip(),
                case=False,
                na=False,
                regex=False,
            )
        ].copy()

    sort_columns = []
    ascending = []

    if price_sort == "Low to High":
        sort_columns.append("discounted_price")
        ascending.append(True)
    elif price_sort == "High to Low":
        sort_columns.append("discounted_price")
        ascending.append(False)

    if rating_sort != "None":
        sort_columns.append("rating")
        ascending.append(rating_sort == "Low to High")

    if sort_columns:
        filtered_df = filtered_df.sort_values(
            sort_columns,
            ascending=ascending,
            na_position="last",
            kind="stable",
        )

    st.caption(f"{len(filtered_df)} smartphones found")

    if filtered_df.empty:
        st.info("No smartphones match your filters.")
    else:
        render_product_grid(filtered_df, "browse")


# =========================================================
# PAGE 2: SMART PURCHASE DECISION
# =========================================================

elif selected_page == "Smart Purchase Decision":
    st.header("Smart Purchase Decision")
    st.write("Find smartphones that match your budget and priorities.")

    budget_text = st.text_input(
        "Maximum budget (₹)",
        placeholder="Enter your budget, e.g. 32000",
        key="decision_budget",
        on_change=clear_smartbuy_decision,
    )

    rating_choice = st.selectbox(
        "Minimum customer rating",
        options=[0.0, 3.0, 3.5, 4.0, 4.5],
        index=None,
        placeholder="Choose a minimum rating",
        format_func=lambda x: (
            "Any rating" if x == 0 else f"{x}+ stars"
        ),
        key="decision_rating",
        on_change=clear_smartbuy_decision,
    )

    st.subheader("Your priorities")

    balanced = st.checkbox(
        "Balanced recommendation",
        key="decision_balanced",
        on_change=clear_smartbuy_decision,
    )

    priority_cols = st.columns(4)
    priority_values = [
        ("Camera", "Camera"),
        ("Battery", "Battery"),
        ("Performance", "Performance"),
        ("Display", "Display"),
    ]

    chosen_priorities = []

    for column, (priority_name, label) in zip(
        priority_cols, priority_values
    ):
        with column:
            chosen = st.checkbox(
                label,
                key=f"decision_{priority_name.lower()}",
                disabled=st.session_state["decision_balanced"],
                on_change=clear_smartbuy_decision,
            )
            if chosen:
                chosen_priorities.append(priority_name)

    submitted = st.button(
        "Get Recommendations",
        type="primary",
        width="stretch",
        key="get_smartbuy_recommendations",
    )

    if submitted:
        st.session_state["smartbuy_decision"] = None

        try:
            if not budget_text.strip():
                raise ValueError("Please enter your budget.")

            budget_value = float(
                budget_text.replace(",", "").replace("₹", "").strip()
            )

            if not math.isfinite(budget_value) or budget_value <= 0:
                raise ValueError("Budget must be greater than ₹0.")

            if rating_choice is None:
                raise ValueError("Please select a minimum rating.")

            priorities = ["Balanced"] if balanced else chosen_priorities

            if not priorities:
                raise ValueError(
                    "Select Balanced or at least one priority."
                )

            priorities = validate_priorities(priorities)

            st.session_state["smartbuy_decision"] = {
                "budget": budget_value,
                "min_rating": float(rating_choice),
                "priorities": priorities,
            }

        except ValueError as error:
            st.session_state["smartbuy_decision"] = None
            st.error(str(error))

    decision = st.session_state.get("smartbuy_decision")

    if decision:
        budget = decision["budget"]
        priorities = decision["priorities"]

        st.subheader("Recommended smartphones")
        st.caption(
            f"Budget: {format_price(budget)} · "
            f"Priority: {get_priority_label(priorities)}"
        )

        try:
            # The recommendation engine requires a positive integer for top_n.
            # Request candidates, then apply the UI filters.
            recommendations = get_recommendations(
                budget=budget,
                priority=priorities,
                min_rating=decision["min_rating"],
                top_n=20,
            )

            if recommendations is None or recommendations.empty:
                st.info(
                    "No matching smartphones found. Try increasing your "
                    "budget or lowering the minimum rating."
                )
            else:
                recommendations = recommendations.copy()

                recommendations["discounted_price"] = pd.to_numeric(
                    recommendations["discounted_price"],
                    errors="coerce",
                )
                recommendations["rating"] = pd.to_numeric(
                    recommendations["rating"],
                    errors="coerce",
                )

                recommendations = recommendations[
                    recommendations["discounted_price"].notna()
                    & recommendations["discounted_price"].gt(0)
                    & recommendations["discounted_price"].le(budget)
                ].copy()

                if decision["min_rating"] > 0:
                    recommendations = recommendations[
                        recommendations["rating"].notna()
                        & recommendations["rating"].ge(
                            decision["min_rating"]
                        )
                    ].copy()

                preferred_floor = get_preferred_price_floor(budget)

                near_budget = recommendations[
                    recommendations["discounted_price"].ge(preferred_floor)
                ].copy()

                if len(near_budget) >= 3:
                    recommendations = near_budget

                recommendations["_score_sort"] = pd.to_numeric(
                    recommendations.get(
                        "recommendation_score",
                        pd.Series(0, index=recommendations.index),
                    ),
                    errors="coerce",
                ).fillna(0)

                recommendations["_rating_sort"] = (
                    recommendations["rating"].fillna(-1)
                )

                recommendations["_budget_fit"] = (
                    recommendations["discounted_price"] / budget
                ).clip(0, 1) * 100

                recommendations = recommendations.sort_values(
                    ["_budget_fit", "_score_sort", "_rating_sort"],
                    ascending=[False, False, False],
                    kind="stable",
                ).drop(
                    columns=[
                        "_score_sort",
                        "_rating_sort",
                        "_budget_fit",
                    ],
                    errors="ignore",
                ).head(5).reset_index(drop=True)

                if recommendations.empty:
                    st.info(
                        "No smartphones match your budget and rating."
                    )
                else:
                    render_recommendation_cards(
                        recommendations, priorities
                    )

        except (
            ValueError,
            FileNotFoundError,
            RuntimeError,
            KeyError,
            TypeError,
        ):
            st.error(
                "We couldn't generate recommendations right now. "
                "Please try different preferences."
            )


# =========================================================
# PAGE 3: REVIEW INSIGHTS & CONFLICT DETECTION
# =========================================================

elif selected_page == "Review Insights & Conflict Detection":
    st.header("Review Insights & Conflict Detection")
    st.write(
        "Explore customer feedback about smartphone features and compare "
        "positive and negative experiences."
    )

    search_review_product = st.text_input(
        "Search smartphones",
        placeholder="Enter a smartphone name...",
        key="review_product_search",
    )

    review_df = df[
        df["review_title"].str.strip().ne("")
        | df["review_content"].str.strip().ne("")
    ].copy()

    if not search_review_product.strip():
        st.info("Search for a smartphone to view its review insights.")
        st.stop()

    review_df = review_df[
        review_df["product_name"].str.contains(
            search_review_product.strip(),
            case=False,
            na=False,
            regex=False,
        )
    ].copy()

    if review_df.empty:
        st.info("No matching smartphone reviews were found.")
        st.stop()

    product_ids = review_df["product_id"].astype(str).tolist()

    name_by_id = {
        str(row["product_id"]): str(row["product_name"])
        for _, row in review_df.iterrows()
    }

    chosen_id = st.selectbox(
        "Choose a smartphone",
        options=product_ids,
        index=None,
        placeholder="Select a matching smartphone",
        format_func=lambda pid: name_by_id.get(str(pid), "Smartphone"),
        key="review_product_choice",
    )

    if chosen_id is None:
        st.info("Choose a smartphone to view its insights.")
        st.stop()

    selected_rows = review_df[
        review_df["product_id"].astype(str) == str(chosen_id)
    ]

    if selected_rows.empty:
        st.info("Choose a valid smartphone.")
        st.stop()

    chosen_product = selected_rows.iloc[0].to_dict()

    image_col, detail_col = st.columns([1, 2], gap="large")

    with image_col:
        show_product_image(chosen_product, width=230, slot_height=230)
        show_price(chosen_product)
        show_rating(chosen_product)
        show_product_link(chosen_product)

    with detail_col:
        st.subheader(
            safe_text(chosen_product.get("product_name"), "Smartphone")
        )
        description = safe_text(chosen_product.get("about_product"))
        if description:
            st.write(description)

    with st.expander("Read customer reviews"):
        cleaned_text = get_review_text(
            chosen_product.get("review_title", ""),
            chosen_product.get("review_content", ""),
        )
        st.write(cleaned_text or "No readable review text available.")

    try:
        results = analyse_product_reviews(chosen_product)
        observations = results["observations"]
        summary = results["summary"]
        conflicts = results["conflicts"]

        if not observations:
            st.info("There isn't enough review text to show insights.")
        else:
            st.subheader("Customer feedback")

            positive_count = sum(
                item["sentiment"] == "Positive"
                for item in observations
            )
            negative_count = sum(
                item["sentiment"] == "Negative"
                for item in observations
            )

            metric_cols = st.columns(3)
            metric_cols[0].metric("Positive mentions", positive_count)
            metric_cols[1].metric("Negative mentions", negative_count)
            metric_cols[2].metric(
                "Features discussed",
                summary["Aspect"].nunique() if not summary.empty else 0,
            )

            st.subheader("Feedback by feature")

            if not summary.empty:
                st.dataframe(
                    summary,
                    hide_index=True,
                    width="stretch",
                )

            aspects = (
                summary["Aspect"].dropna().astype(str).tolist()
                if not summary.empty else []
            )

            if aspects:
                chosen_aspect = st.selectbox(
                    "Explore a feature",
                    options=aspects,
                    key="review_aspect_choice",
                )

                aspect_items = [
                    item for item in observations
                    if item["aspect"] == chosen_aspect
                ]

                groups = {
                    "Positive": [],
                    "Negative": [],
                    "Mixed": [],
                    "Neutral": [],
                }

                for item in aspect_items:
                    sentiment = item["sentiment"]
                    sentence = item["sentence"]

                    if (
                        sentiment in groups
                        and sentence not in groups[sentiment]
                    ):
                        groups[sentiment].append(sentence)

                positive_col, negative_col = st.columns(2)

                with positive_col:
                    st.markdown("**Positive feedback**")
                    if groups["Positive"]:
                        for sentence in groups["Positive"]:
                            st.markdown(f"- {sentence}")
                    else:
                        st.caption("No positive feedback identified.")

                with negative_col:
                    st.markdown("**Negative feedback**")
                    if groups["Negative"]:
                        for sentence in groups["Negative"]:
                            st.markdown(f"- {sentence}")
                    else:
                        st.caption("No negative feedback identified.")

                other_feedback = groups["Mixed"] + groups["Neutral"]

                if other_feedback:
                    with st.expander("Other feedback"):
                        for sentence in other_feedback:
                            st.markdown(f"- {sentence}")

            st.subheader("Conflicting opinions")

            if conflicts.empty:
                st.info(
                    "No conflicting positive and negative feedback "
                    "was identified."
                )
            else:
                for _, conflict in conflicts.iterrows():
                    with st.expander(
                        f"{conflict['Aspect']}: Mixed experiences"
                    ):
                        positive_col, negative_col = st.columns(2)

                        with positive_col:
                            st.markdown("**Positive feedback**")
                            for sentence in conflict["Positive evidence"]:
                                st.markdown(f"- {sentence}")

                        with negative_col:
                            st.markdown("**Negative feedback**")
                            for sentence in conflict["Negative evidence"]:
                                st.markdown(f"- {sentence}")

            st.caption(
                "Review insights are intended as guidance. "
                "Please check the original reviews before making a decision."
            )

    except (
        KeyError,
        ValueError,
        TypeError,
        RuntimeError,
        AttributeError,
    ):
        st.error(
            "We couldn't analyse these reviews right now. "
            "Please try another smartphone."
        )


# =========================================================
# PAGE 4: BUDGET ALTERNATIVES & BEST VALUE
# =========================================================

elif selected_page == "Budget Alternatives & Best Value":
    st.header("Budget Alternatives & Best Value")
    st.write(
        "Compare smartphones that fit your budget using customer ratings "
        "and estimated value."
    )

    with st.form("combined_comparison_form"):
        budget_text = st.text_input(
            "Maximum budget (₹)",
            placeholder="For example, 32000",
            key="combined_budget",
        )

        rating_choice = st.selectbox(
            "Minimum customer rating",
            options=[0.0, 3.0, 3.5, 4.0, 4.5],
            index=0,
            format_func=lambda x: (
                "Any rating" if x == 0 else f"{x}+ stars"
            ),
            key="combined_rating",
        )

        submitted = st.form_submit_button(
            "Compare Alternatives & Value",
            type="primary",
            width="stretch",
        )

    if submitted:
        st.session_state["combined_comparison_result"] = None

        try:
            if not budget_text.strip():
                raise ValueError("Please enter your maximum budget.")

            budget = float(
                budget_text.replace(",", "").replace("₹", "").strip()
            )

            if not math.isfinite(budget) or budget <= 0:
                raise ValueError("Budget must be greater than ₹0.")

            minimum_rating = float(rating_choice)

            eligible, effective_floor, expanded = (
                filter_products_for_budget_band(
                    df,
                    budget,
                    min_rating=minimum_rating,
                    minimum_products=3,
                )
            )

            if eligible.empty:
                st.info(
                    "No smartphones match your budget and rating. "
                    "Try increasing your budget or lowering the rating."
                )
            else:
                value_df = calculate_value_scores(
                    products=eligible,
                    budget=budget,
                    min_rating=minimum_rating,
                    top_n=20,
                    use_budget_band=False,
                )

                if value_df is None or value_df.empty:
                    st.info(
                        "No matching smartphones could be compared. "
                        "Try adjusting your preferences."
                    )

                elif "value_score" not in value_df.columns:
                    st.error(
                        "We couldn't calculate the comparison at this time."
                    )

                else:
                    value_df = value_df.copy()

                    value_df["discounted_price"] = pd.to_numeric(
                        value_df["discounted_price"],
                        errors="coerce",
                    )
                    value_df["value_score"] = pd.to_numeric(
                        value_df["value_score"],
                        errors="coerce",
                    )

                    value_df = value_df[
                        value_df["discounted_price"].notna()
                        & value_df["discounted_price"].gt(0)
                        & value_df["discounted_price"].le(budget)
                        & value_df["discounted_price"].ge(effective_floor)
                        & value_df["value_score"].notna()
                    ].copy()

                    if value_df.empty:
                        st.info(
                            "No smartphones match these comparison settings."
                        )
                    else:
                        value_df = add_display_value_score(value_df)

                        value_df["rating_sort"] = pd.to_numeric(
                            value_df["rating"],
                            errors="coerce",
                        ).fillna(-1)

                        value_df = value_df.sort_values(
                            [
                                "display_value_score",
                                "rating_sort",
                                "discounted_price",
                            ],
                            ascending=[False, False, True],
                            na_position="last",
                            kind="stable",
                        ).drop(
                            columns=["rating_sort"],
                            errors="ignore",
                        )

                        st.session_state["combined_comparison_result"] = {
                            "budget": budget,
                            "products": value_df.reset_index(drop=True),
                        }

        except (
            ValueError,
            TypeError,
            FileNotFoundError,
            RuntimeError,
            KeyError,
        ):
            st.session_state["combined_comparison_result"] = None
            st.error(
                "We couldn't compare smartphones with those settings. "
                "Please check your budget and try again."
            )

    result = st.session_state.get("combined_comparison_result")

    if result is not None:
        products = result["products"]
        budget = result["budget"]

        if products.empty:
            st.info("No smartphones are available for this comparison.")
        else:
            st.subheader("Your smartphone comparison")

            metric_cols = st.columns(2)

            metric_cols[0].metric(
                "Matching smartphones",
                len(products),
            )

            metric_cols[1].metric(
                "Highest value score",
                f"{products['display_value_score'].max():.1f}/100",
            )

            render_comparison_cards(products.head(12), budget)

            if len(products) > 12:
                st.caption("Showing the top 12 matching smartphones.")


# =========================================================
# PAGE 5: PURCHASE REGRET PREVENTION
# =========================================================

elif selected_page == "Purchase Regret Prevention":
    st.header("Purchase Regret Prevention")
    st.caption(
        "Check whether a smartphone fits your budget and review your "
        "potential concerns before buying."
    )

    budget_text = st.text_input(
        "Maximum budget (₹)",
        placeholder="e.g. 32000",
        key="risk_budget",
    )

    budget = None
    budget_valid = False

    if budget_text.strip():
        try:
            budget = float(
                budget_text.replace(",", "").replace("₹", "").strip()
            )

            if not math.isfinite(budget) or budget <= 0:
                raise ValueError

            budget_valid = True

        except ValueError:
            st.error("Enter a valid budget greater than ₹0.")

    if not budget_valid:
        st.info("Enter a valid maximum budget to continue.")
        st.stop()

    search = st.text_input(
        "Search smartphone",
        placeholder="Enter a phone name or model...",
        key="risk_search",
    )

    if not search.strip():
        st.info("Search for a smartphone after entering your budget.")
        st.stop()

    matches = df[
        df["product_name"].str.contains(
            search.strip(),
            case=False,
            na=False,
            regex=False,
        )
    ].copy()

    if matches.empty:
        st.warning("No matching smartphone found.")
        st.stop()

    products = {
        str(row["product_id"]): row.to_dict()
        for _, row in matches.sort_values("product_name").iterrows()
    }

    selected_id = st.selectbox(
        "Choose smartphone",
        options=list(products.keys()),
        index=None,
        placeholder="Select a smartphone",
        format_func=lambda pid: (
            f"{products[pid]['product_name']} — "
            f"{format_price(products[pid]['discounted_price'])}"
        ),
        key="risk_phone",
    )

    if selected_id is None:
        st.info("Select a smartphone to see its purchase assessment.")
        st.stop()

    product = products[selected_id]
    phone_price = float(product["discounted_price"])

    image_col, details_col = st.columns([1, 2])

    with image_col:
        show_product_image(product, width=200, slot_height=200)

    with details_col:
        st.subheader(safe_text(product.get("product_name"), "Smartphone"))
        st.markdown(f"### {format_price(phone_price)}")
        show_rating(product)
        show_product_link(product)

    if phone_price <= budget:
        st.success(
            f"Within budget — "
            f"{format_price(budget - phone_price)} remaining."
        )
    else:
        st.warning(
            f"Over budget by {format_price(phone_price - budget)}."
        )

    try:
        review_results = analyse_product_reviews(product)
        observations = review_results.get("observations", [])
        conflicts = review_results.get("conflicts", pd.DataFrame())

    except (
        KeyError,
        ValueError,
        TypeError,
        RuntimeError,
        AttributeError,
    ):
        observations = []
        conflicts = pd.DataFrame()

    negative_observations = [
        item for item in observations
        if item.get("sentiment") == "Negative"
    ]

    aspect_concerns = {}

    for item in negative_observations:
        aspect = safe_text(item.get("aspect"), "Customer feedback")
        sentence = safe_text(item.get("sentence"), "")

        if sentence and aspect not in aspect_concerns:
            aspect_concerns[aspect] = sentence

    review_text = get_review_text(
        product.get("review_title", ""),
        product.get("review_content", ""),
    )

    st.subheader("Think before you buy")

    concerns = []

    if phone_price > budget:
        concerns.append((
            "Over your budget",
            f"This phone costs {format_price(phone_price - budget)} "
            "more than you planned.",
            "Consider whether the extra cost is worthwhile for you.",
        ))

    for aspect, sentence in aspect_concerns.items():
        concerns.append((
            f"{aspect} concerns",
            sentence,
            f"Consider whether {aspect.lower()} is important to you.",
        ))

        if len(concerns) >= 3:
            break

    if (not review_text.strip() or not observations) and len(concerns) < 3:
        concerns.append((
            "Limited review information",
            "There isn't enough readable review feedback to assess "
            "this phone confidently.",
            "Check recent customer reviews before purchasing.",
        ))

    if not concerns:
        st.success(
            "No major concerns were identified from the available information. "
            "This does not guarantee a problem-free purchase."
        )
    else:
        for title, detail, impact in concerns[:3]:
            with st.expander(title, expanded=False):
                st.write(detail)
                st.caption(impact)

    if not conflicts.empty and len(concerns) < 3:
        aspects = conflicts["Aspect"].dropna().astype(str).tolist()

        if aspects:
            with st.expander("Conflicting customer opinions"):
                st.write(
                    "Customer opinions differ on "
                    + ", ".join(aspects[:2])
                    + ". Review the feedback to decide whether these "
                    "trade-offs matter to you."
                )

    st.caption(
        "This assessment highlights possible concerns based on available "
        "product and review information. It cannot guarantee future outcomes."
    )