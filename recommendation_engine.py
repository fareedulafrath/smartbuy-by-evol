import math
import sqlite3
import difflib
from contextlib import closing
from pathlib import Path

import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "shopping_assistant.db"

OTHER_PRIORITIES = ["Camera", "Battery", "Performance", "Display"]

REQUIRED_COLUMNS = {
    "product_id",
    "product_name",
    "discounted_price",
    "rating",
    "about_product",
}

PRIORITY_WEIGHTS = {
    "Balanced": {
        "rating_score": 0.70,
        "camera_score": 0.10,
        "battery_score": 0.10,
        "performance_score": 0.05,
        "display_score": 0.05,
    },
    "Camera": {
        "rating_score": 0.30,
        "camera_score": 0.55,
        "battery_score": 0.05,
        "performance_score": 0.05,
        "display_score": 0.05,
    },
    "Battery": {
        "rating_score": 0.30,
        "camera_score": 0.05,
        "battery_score": 0.55,
        "performance_score": 0.05,
        "display_score": 0.05,
    },
    "Performance": {
        "rating_score": 0.30,
        "camera_score": 0.05,
        "battery_score": 0.05,
        "performance_score": 0.55,
        "display_score": 0.05,
    },
    "Display": {
        "rating_score": 0.30,
        "camera_score": 0.05,
        "battery_score": 0.05,
        "performance_score": 0.05,
        "display_score": 0.55,
    },
}

DEFAULT_BUDGET_FLOOR_RATIO = 0.60
MIN_PREFERRED_RESULTS = 3
FALLBACK_BUDGET_FLOOR_RATIO = 0.40

VALUE_WEIGHTS = {
    "rating_component": 0.35,
    "specification_component": 0.45,
    "price_component": 0.20,
}

SPECIFICATION_COLUMNS = [
    "camera_score",
    "battery_score",
    "performance_score",
    "display_score",
]


# =========================================================
# VALIDATION
# =========================================================

def validate_number(value, name, minimum, maximum=None):
    """Validate a finite numeric value. Inputs such as '4+' are accepted."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a valid number.")

    try:
        cleaned = str(value).strip()

        if cleaned.endswith("+"):
            cleaned = cleaned[:-1].strip()

        number = float(cleaned)

    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a valid number.") from None

    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")

    if number < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")

    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must not exceed {maximum}.")

    return number


def validate_top_n(top_n):
    """Require a positive integer."""

    if isinstance(top_n, bool):
        raise ValueError("top_n must be a positive integer.")

    try:
        number = float(top_n)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("top_n must be a positive integer.") from None

    if not math.isfinite(number) or number < 1 or not number.is_integer():
        raise ValueError("top_n must be a positive integer.")

    return int(number)


def validate_priorities(priorities):
    """Validate Balanced or one/two specific shopping priorities."""

    if isinstance(priorities, str):
        raw_priorities = [priorities]
    elif isinstance(priorities, (list, tuple, set)):
        raw_priorities = list(priorities)
    else:
        raise ValueError(
            "Choose Balanced or one/two priorities from "
            "Camera, Battery, Performance, Display."
        )

    if not raw_priorities:
        raise ValueError("Please select at least one priority.")

    valid_names = ["Balanced"] + OTHER_PRIORITIES
    normalized_priorities = []

    for priority in raw_priorities:
        if not isinstance(priority, str) or not priority.strip():
            raise ValueError("Each priority must be a valid name.")

        normalized = priority.strip().casefold()

        exact_match = next(
            (
                name for name in valid_names
                if name.casefold() == normalized
            ),
            None,
        )

        if exact_match is None:
            matches = difflib.get_close_matches(
                normalized,
                [name.casefold() for name in valid_names],
                n=1,
                cutoff=0.75,
            )

            if matches:
                exact_match = next(
                    name for name in valid_names
                    if name.casefold() == matches[0]
                )

        if exact_match is None:
            raise ValueError(
                f"Unknown priority '{priority}'. Choose from "
                "Balanced, Camera, Battery, Performance, Display."
            )

        if exact_match not in normalized_priorities:
            normalized_priorities.append(exact_match)

    if "Balanced" in normalized_priorities:
        if len(normalized_priorities) != 1:
            raise ValueError(
                "Balanced is a standalone option. "
                "Do not combine it with other priorities."
            )
        return ["Balanced"]

    if len(normalized_priorities) > 2:
        raise ValueError("Please choose no more than two priorities.")

    return [
        name for name in OTHER_PRIORITIES
        if name in normalized_priorities
    ]


def get_priority_label(priorities):
    """Return a readable label for the selected priorities."""

    return " + ".join(validate_priorities(priorities))


# =========================================================
# DATABASE LOADING
# =========================================================

def load_products():
    """Read and validate product data from SQLite without modifying it."""

    if not DB_PATH.is_file():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}. "
            "Run setup_database.py first."
        )

    database_uri = DB_PATH.as_uri() + "?mode=ro"

    try:
        with closing(
            sqlite3.connect(
                database_uri,
                uri=True,
                timeout=10,
            )
        ) as connection:
            products = pd.read_sql_query(
                "SELECT * FROM products",
                connection,
            )

    except (sqlite3.Error, pd.errors.DatabaseError) as error:
        raise RuntimeError(
            "Could not read the products table. "
            "Check that the database and products table exist."
        ) from error

    missing = REQUIRED_COLUMNS - set(products.columns)

    if missing:
        raise ValueError(
            "Missing database columns: " + ", ".join(sorted(missing))
        )

    if products.empty:
        raise ValueError("The products table contains no products.")

    products["product_id"] = (
        products["product_id"].astype("string").str.strip()
    )
    products["product_name"] = (
        products["product_name"].astype("string").str.strip()
    )

    products["discounted_price"] = pd.to_numeric(
        products["discounted_price"],
        errors="coerce",
    )
    products["rating"] = pd.to_numeric(
        products["rating"],
        errors="coerce",
    )

    valid_prices = (
        products["discounted_price"].notna()
        & products["discounted_price"].map(
            lambda value: math.isfinite(float(value))
        )
        & products["discounted_price"].ge(0)
    )

    valid_products = (
        products["product_id"].notna()
        & products["product_id"].ne("")
        & products["product_name"].notna()
        & products["product_name"].ne("")
        & valid_prices
    )

    products = products.loc[valid_products].copy()

    valid_ratings = (
        products["rating"].notna()
        & products["rating"].map(
            lambda value: math.isfinite(float(value))
        )
        & products["rating"].between(0, 5)
    )

    products.loc[~valid_ratings, "rating"] = float("nan")

    products = products.drop_duplicates(
        subset=["product_id"],
        keep="first",
    ).copy()

    products["about_product"] = (
        products["about_product"].fillna("").astype(str)
    )

    optional_defaults = {
        "actual_price": float("nan"),
        "rating_count": float("nan"),
        "discount_percentage": float("nan"),
        "review_title": "",
        "review_content": "",
        "img_link": "",
        "product_link": "",
    }

    for column, default in optional_defaults.items():
        if column not in products.columns:
            products[column] = default

    for column in (
        "review_title",
        "review_content",
        "img_link",
        "product_link",
    ):
        products[column] = products[column].fillna("").astype(str)

    if products.empty:
        raise ValueError("No valid products remain after validation.")

    return products.reset_index(drop=True)


# =========================================================
# SPECIFICATION EXTRACTION
# =========================================================

def add_specification_scores(products):
    """
    Extract rough specification indicators from product text.

    These are text-based heuristics, not independent hardware benchmarks.
    Missing evidence remains unknown rather than automatically scoring zero.
    """

    products = products.copy()

    names = (
        products["product_name"].fillna("").astype(str)
        if "product_name" in products.columns
        else pd.Series("", index=products.index)
    )

    descriptions = (
        products["about_product"].fillna("").astype(str)
        if "about_product" in products.columns
        else pd.Series("", index=products.index)
    )

    specs = (names + " " + descriptions).str.lower()

    # Camera: largest megapixel number mentioned.
    mp_values = specs.str.findall(r"(\d+(?:\.\d+)?)\s*mp\b")

    def largest_mp(values):
        numbers = []

        for value in values:
            try:
                number = float(value)
                if math.isfinite(number) and number > 0:
                    numbers.append(number)
            except (TypeError, ValueError, OverflowError):
                continue

        return max(numbers) if numbers else float("nan")

    camera_mp = mp_values.map(largest_mp)
    products["camera_score"] = camera_mp.clip(upper=200) / 200 * 100

    # Battery capacity in mAh.
    battery_values = specs.str.extract(
        r"\b(\d{3,5})\s*mah\b",
        expand=False,
    )

    battery_mah = pd.to_numeric(battery_values, errors="coerce")
    battery_mah = battery_mah.where(battery_mah.between(1000, 20000))

    products["battery_score"] = (
        battery_mah.clip(upper=10000) / 10000 * 100
    )

    # Processor-family presence is only a weak indicator.
    processor_pattern = (
        r"snapdragon|dimensity|mediatek|exynos|"
        r"helio|unisoc|kirin|tensor|bionic"
    )

    has_processor = specs.str.contains(
        processor_pattern,
        regex=True,
        na=False,
    )

    products["performance_score"] = float("nan")
    products.loc[has_processor, "performance_score"] = 100.0

    # Display information is a presence indicator, not a quality ranking.
    display_pattern = r"amoled|oled|\b(?:60|90|120|144|165)\s*hz\b"

    has_display_info = specs.str.contains(
        display_pattern,
        regex=True,
        na=False,
    )

    products["display_score"] = float("nan")
    products.loc[has_display_info, "display_score"] = 100.0

    # Customer rating normalized to 0–100.
    if "rating" in products.columns:
        products["rating"] = pd.to_numeric(
            products["rating"],
            errors="coerce",
        )

        products["rating_score"] = (
            products["rating"].where(
                products["rating"].between(0, 5)
            ) / 5 * 100
        )
    else:
        products["rating_score"] = float("nan")

    # RAM: "8GB RAM", "8 GB RAM", or "RAM: 8GB".
    ram_values = specs.str.extract(
        r"\b(\d{1,3})\s*gb\s*ram\b"
        r"|\bram\s*:?\s*(\d{1,3})\s*gb\b",
        expand=True,
    )

    products["ram_gb"] = pd.to_numeric(
        ram_values[0].fillna(ram_values[1]),
        errors="coerce",
    )

    products.loc[
        ~products["ram_gb"].between(1, 128),
        "ram_gb",
    ] = float("nan")

    return products


# =========================================================
# SHARED BUDGET-BAND FILTERING
# =========================================================

def filter_budget_band(
    products,
    budget,
    min_rating=0.0,
    floor_ratio=DEFAULT_BUDGET_FLOOR_RATIO,
    minimum_results=MIN_PREFERRED_RESULTS,
    fallback_floor_ratio=FALLBACK_BUDGET_FLOOR_RATIO,
):
    """
    Filter products to a useful spending range below the user's budget.

    First try the preferred price band. If too few products qualify,
    expand the lower price boundary. Never include products above budget.
    """

    budget = validate_number(budget, "Budget", minimum=0.01)
    min_rating = validate_number(
        min_rating, "Minimum rating", minimum=0, maximum=5
    )
    floor_ratio = validate_number(
        floor_ratio, "Price floor ratio", minimum=0.01, maximum=1
    )
    fallback_floor_ratio = validate_number(
        fallback_floor_ratio,
        "Fallback price floor ratio",
        minimum=0.01,
        maximum=1,
    )

    if fallback_floor_ratio > floor_ratio:
        raise ValueError(
            "Fallback price floor must not exceed the preferred floor."
        )

    if (
        isinstance(minimum_results, bool)
        or not isinstance(minimum_results, int)
        or minimum_results < 1
    ):
        raise ValueError("minimum_results must be a positive integer.")

    if products is None:
        return pd.DataFrame()

    if products.empty:
        return products.copy()

    candidates = products.copy()

    for column in ("discounted_price", "rating"):
        if column not in candidates.columns:
            candidates[column] = float("nan")
        candidates[column] = pd.to_numeric(
            candidates[column],
            errors="coerce",
        )

    valid_price = (
        candidates["discounted_price"].notna()
        & candidates["discounted_price"].gt(0)
        & candidates["discounted_price"].map(
            lambda value: math.isfinite(float(value))
        )
        & candidates["discounted_price"].le(budget)
    )

    # Missing ratings are allowed when the user has not requested
    # a minimum rating. They cannot qualify for a positive threshold.
    if min_rating == 0:
        valid_rating = (
            candidates["rating"].isna()
            | candidates["rating"].between(0, 5)
        )
    else:
        valid_rating = (
            candidates["rating"].notna()
            & candidates["rating"].between(min_rating, 5)
        )

    eligible = candidates.loc[valid_price & valid_rating].copy()

    preferred_floor = budget * floor_ratio
    fallback_floor = budget * fallback_floor_ratio

    preferred = eligible.loc[
        eligible["discounted_price"].ge(preferred_floor)
    ].copy()

    if len(preferred) >= minimum_results:
        result = preferred
        actual_floor = preferred_floor
        range_status = "preferred"
    else:
        fallback = eligible.loc[
            eligible["discounted_price"].ge(fallback_floor)
        ].copy()

        if len(fallback) >= minimum_results:
            result = fallback
            actual_floor = fallback_floor
            range_status = "expanded"
        else:
            result = eligible.copy()
            actual_floor = 0.0
            range_status = "all_under_budget"

    result = result.copy()
    result["comparison_budget"] = budget
    result["comparison_price_floor"] = actual_floor
    result["comparison_price_ceiling"] = budget
    result["comparison_range_status"] = range_status

    result.attrs["budget"] = budget
    result.attrs["price_floor"] = actual_floor
    result.attrs["price_ceiling"] = budget
    result.attrs["range_status"] = range_status

    return result.reset_index(drop=True)


def get_budget_alternatives(
    products,
    budget,
    min_rating=0.0,
    top_n=12,
):
    """Rank affordable alternatives by rating, then lower price."""

    top_n = validate_top_n(top_n)

    candidates = filter_budget_band(
        products=products,
        budget=budget,
        min_rating=min_rating,
    )

    if candidates.empty:
        return candidates

    candidates = candidates.sort_values(
        ["rating", "discounted_price"],
        ascending=[False, True],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)

    return candidates.head(top_n).reset_index(drop=True)


# =========================================================
# SMART PURCHASE DECISION
# =========================================================

def get_recommendations(
    budget,
    priority="Balanced",
    min_rating=0.0,
    top_n=5,
):
    """Rank eligible phones by the selected shopping priorities."""

    budget = validate_number(budget, "Budget", minimum=0.01)
    min_rating = validate_number(
        min_rating, "Minimum rating", minimum=0, maximum=5
    )
    priorities = validate_priorities(priority)
    top_n = validate_top_n(top_n)

    products = load_products()

    candidates = products.loc[
        products["discounted_price"].le(budget)
        & products["rating"].notna()
        & products["rating"].ge(min_rating)
        & products["discounted_price"].gt(0)
    ].copy()

    if candidates.empty:
        return candidates.reset_index(drop=True)

    candidates = add_specification_scores(candidates)

    feature_columns = {
        "Camera": "camera_score",
        "Battery": "battery_score",
        "Performance": "performance_score",
        "Display": "display_score",
    }

    if priorities == ["Balanced"]:
        weights = PRIORITY_WEIGHTS["Balanced"]
    else:
        weights = {"rating_score": 0.30}
        per_priority_weight = 0.70 / len(priorities)

        for selected_priority in priorities:
            weights[feature_columns[selected_priority]] = per_priority_weight

        for selected_priority in priorities:
            feature = feature_columns[selected_priority]
            candidates = candidates.loc[candidates[feature].notna()].copy()

        if candidates.empty:
            return candidates.reset_index(drop=True)

    weighted_scores = pd.Series(0.0, index=candidates.index, dtype=float)
    available_weights = pd.Series(0.0, index=candidates.index, dtype=float)

    for column, weight in weights.items():
        available = candidates[column].notna()

        weighted_scores.loc[available] += (
            candidates.loc[available, column] * weight
        )
        available_weights.loc[available] += weight

    has_evidence = available_weights > 0

    candidates = candidates.loc[has_evidence].copy()
    weighted_scores = weighted_scores.loc[has_evidence]
    available_weights = available_weights.loc[has_evidence]

    if candidates.empty:
        return candidates.reset_index(drop=True)

    candidates["recommendation_score"] = (
        weighted_scores / available_weights
    ).clip(0, 100).round(1)

    candidates["evidence_coverage"] = (
        available_weights / sum(weights.values()) * 100
    ).round(0).astype(int)

    candidates = candidates.sort_values(
        by=[
            "recommendation_score",
            "rating",
            "ram_gb",
            "discounted_price",
        ],
        ascending=[False, False, False, True],
        na_position="last",
        kind="stable",
    )

    return candidates.head(top_n).reset_index(drop=True)


def explain_recommendation(product, priority="Balanced"):
    """Explain a Smart Purchase Decision recommendation."""

    priorities = validate_priorities(priority)
    priority_label = get_priority_label(priorities)

    name = str(product.get("product_name", "This smartphone"))
    price = float(product["discounted_price"])
    rating = product.get("rating")
    score = product.get("recommendation_score")
    coverage = product.get("evidence_coverage")

    reasons = [f"{name} is priced at ₹{price:,.0f}."]

    if pd.notna(rating):
        reasons.append(
            f"Its listed customer rating is {float(rating):.1f}/5."
        )

    if pd.notna(score):
        reasons.append(
            f"Its heuristic {priority_label.lower()} score is "
            f"{float(score):.1f}/100."
        )

    if pd.notna(coverage):
        reasons.append(
            f"Scoring evidence coverage is {int(coverage)}%."
        )

    reasons.append(
        "The score uses available product text and ratings. "
        "It is not an independent hardware benchmark."
    )

    return " ".join(reasons)


# =========================================================
# BEST VALUE FOR MONEY
# =========================================================

def calculate_value_scores(
    products,
    budget=None,
    min_rating=0.0,
    top_n=None,
    floor_ratio=DEFAULT_BUDGET_FLOOR_RATIO,
    use_budget_band=True,
):
    """
    Calculate estimated value scores for products.

    The score combines:
      - Customer rating: 35%
      - Available specification indicators: 45%
      - Relative affordability: 20%

    Weights are redistributed over components that have usable evidence.
    The resulting score is a comparison heuristic, not a benchmark.
    """

    min_rating = validate_number(
        min_rating, "Minimum rating", minimum=0, maximum=5
    )

    if top_n is not None:
        top_n = validate_top_n(top_n)

    if products is None:
        return pd.DataFrame()

    if products.empty:
        return products.copy()

    candidates = products.copy()

    if budget is not None:
        budget = validate_number(budget, "Budget", minimum=0.01)

        if use_budget_band:
            candidates = filter_budget_band(
                candidates,
                budget=budget,
                min_rating=min_rating,
                floor_ratio=floor_ratio,
            )
        else:
            candidates["discounted_price"] = pd.to_numeric(
                candidates["discounted_price"],
                errors="coerce",
            )
            candidates["rating"] = pd.to_numeric(
                candidates["rating"],
                errors="coerce",
            )

            valid_price = (
                candidates["discounted_price"].notna()
                & candidates["discounted_price"].gt(0)
                & candidates["discounted_price"].le(budget)
            )

            if min_rating > 0:
                valid_rating = (
                    candidates["rating"].notna()
                    & candidates["rating"].between(min_rating, 5)
                )
            else:
                valid_rating = (
                    candidates["rating"].isna()
                    | candidates["rating"].between(0, 5)
                )

            candidates = candidates.loc[
                valid_price & valid_rating
            ].copy()

    else:
        candidates["discounted_price"] = pd.to_numeric(
            candidates["discounted_price"],
            errors="coerce",
        )
        candidates["rating"] = pd.to_numeric(
            candidates["rating"],
            errors="coerce",
        )

        valid_price = (
            candidates["discounted_price"].notna()
            & candidates["discounted_price"].gt(0)
        )

        if min_rating > 0:
            valid_rating = (
                candidates["rating"].notna()
                & candidates["rating"].between(min_rating, 5)
            )
        else:
            valid_rating = (
                candidates["rating"].isna()
                | candidates["rating"].between(0, 5)
            )

        candidates = candidates.loc[valid_price & valid_rating].copy()

    if candidates.empty:
        return candidates.reset_index(drop=True)

    candidates = add_specification_scores(candidates)

    # Component 1: normalized customer rating.
    candidates["rating_component"] = (
        candidates["rating"].where(
            candidates["rating"].between(0, 5)
        ) / 5 * 100
    )

    # Component 2: mean of available specification indicators.
    available_spec_columns = [
        column for column in SPECIFICATION_COLUMNS
        if column in candidates.columns
    ]

    if available_spec_columns:
        candidates["specification_component"] = (
            candidates[available_spec_columns].mean(axis=1, skipna=True)
        )

        candidates["specification_coverage"] = (
            candidates[available_spec_columns].notna().sum(axis=1)
            / len(available_spec_columns)
            * 100
        ).round(0)
    else:
        candidates["specification_component"] = float("nan")
        candidates["specification_coverage"] = 0.0

    # Component 3: relative affordability within this eligible set.
    prices = pd.to_numeric(
        candidates["discounted_price"],
        errors="coerce",
    )

    if budget is not None:
        # A phone near the user's maximum budget receives a lower
        # affordability score than a less expensive eligible phone.
        candidates["price_component"] = (
            (budget - prices).clip(lower=0) / budget * 100
        ).clip(0, 100)
    else:
        minimum_price = prices.min()
        maximum_price = prices.max()

        if (
            pd.notna(minimum_price)
            and pd.notna(maximum_price)
            and maximum_price > minimum_price
        ):
            candidates["price_component"] = (
                (maximum_price - prices)
                / (maximum_price - minimum_price)
                * 100
            ).clip(0, 100)
        else:
            candidates["price_component"] = 100.0

    components = [
        ("rating_component", VALUE_WEIGHTS["rating_component"]),
        (
            "specification_component",
            VALUE_WEIGHTS["specification_component"],
        ),
        ("price_component", VALUE_WEIGHTS["price_component"]),
    ]

    weighted_total = pd.Series(
        0.0,
        index=candidates.index,
        dtype=float,
    )
    available_weight = pd.Series(
        0.0,
        index=candidates.index,
        dtype=float,
    )

    for column, weight in components:
        if column not in candidates.columns:
            continue

        values = pd.to_numeric(candidates[column], errors="coerce")
        available = values.notna() & values.map(
            lambda value: math.isfinite(float(value))
        )

        weighted_total.loc[available] += values.loc[available] * weight
        available_weight.loc[available] += weight

    has_evidence = available_weight > 0
    candidates = candidates.loc[has_evidence].copy()
    weighted_total = weighted_total.loc[has_evidence]
    available_weight = available_weight.loc[has_evidence]

    if candidates.empty:
        return candidates.reset_index(drop=True)

    candidates["value_score"] = (
        weighted_total / available_weight
    ).clip(0, 100).round(1)

    candidates["evidence_coverage"] = (
        available_weight / sum(VALUE_WEIGHTS.values()) * 100
    ).round(0)

    candidates["value_reason"] = candidates.apply(
        explain_value_score,
        axis=1,
    )

    candidates = candidates.sort_values(
        ["value_score", "rating", "discounted_price"],
        ascending=[False, False, True],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)

    if top_n is not None:
        candidates = candidates.head(top_n).reset_index(drop=True)

    return candidates


def explain_value_score(product):
    """Explain an estimated value score using available evidence."""

    name = str(product.get("product_name", "This smartphone"))
    score = product.get("value_score")
    rating = product.get("rating")
    price = product.get("discounted_price")

    reasons = []

    if pd.notna(score):
        reasons.append(f"Estimated value score: {float(score):.1f}/100.")

    if pd.notna(rating):
        reasons.append(f"Customer rating: {float(rating):.1f}/5.")

    if pd.notna(price):
        reasons.append(f"Listed price: ₹{float(price):,.0f}.")

    spec_coverage = product.get("specification_coverage")
    if pd.notna(spec_coverage):
        reasons.append(
            f"Specification evidence coverage: "
            f"{float(spec_coverage):.0f}%."
        )

    reasons.append(
        "This is a relative comparison using available product data, "
        "not an independent hardware benchmark."
    )

    if not reasons:
        return f"No scoring explanation is available for {name}."

    return " ".join(reasons)


# =========================================================
# OPTIONAL COMMAND-LINE CHECK
# =========================================================

if __name__ == "__main__":
    try:
        all_products = load_products()
        print(f"Loaded {len(all_products)} valid products.")

        scored_products = calculate_value_scores(
            products=all_products,
            top_n=5,
        )

        print("\nTop estimated-value products:")
        if scored_products.empty:
            print("No products could be scored.")
        else:
            columns = [
                "product_name",
                "discounted_price",
                "rating",
                "value_score",
            ]
            columns = [
                column for column in columns
                if column in scored_products.columns
            ]
            print(scored_products[columns].to_string(index=False))

    except (FileNotFoundError, ValueError, RuntimeError, sqlite3.Error) as error:
        print(f"Recommendation engine check failed: {error}")