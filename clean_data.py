
import pandas as pd
from pathlib import Path

# STEP 1: Find the project and data folders
project_folder = Path(__file__).parent
data_folder = project_folder / "data"

# STEP 2: Read the original CSV file
input_file = data_folder / "amazon.csv"
df = pd.read_csv(input_file)

print("Original dataset loaded successfully!")
print("Total rows:", len(df))

# STEP 3: Select smartphone products only
df = df[
    df["category"]
    .astype(str)
    .str.endswith("|Smartphones")
].copy()

print("Smartphone rows found:", len(df))

# STEP 4: Remove repeated product IDs
df = df.drop_duplicates(
    subset="product_id",
    keep="first"
)

print("Unique products:", len(df))

# STEP 5: Keep the columns needed for our project
columns_to_keep = [
    "product_id",
    "product_name",
    "category",
    "discounted_price",
    "actual_price",
    "discount_percentage",
    "rating",
    "rating_count",
    "about_product",
    "review_title",
    "review_content",
    "img_link",
    "product_link"
]

# Keep only columns that exist in the CSV
available_columns = [
    col for col in columns_to_keep
    if col in df.columns
]

df = df[available_columns].copy()

# STEP 6: Clean prices
def clean_price(value):
    if pd.isna(value):
        return None

    value = str(value)
    value = value.replace("₹", "")
    value = value.replace(",", "")
    value = value.strip()

    try:
        return float(value)
    except ValueError:
        return None

df["discounted_price"] = df["discounted_price"].apply(clean_price)
df["actual_price"] = df["actual_price"].apply(clean_price)

# STEP 7: Convert numerical columns into numbers
df["rating"] = pd.to_numeric(
    df["rating"], errors="coerce"
)

df["rating_count"] = pd.to_numeric(
    df["rating_count"].astype(str)
    .str.replace(",", "", regex=False),
    errors="coerce"
)

df["discount_percentage"] = pd.to_numeric(
    df["discount_percentage"].astype(str)
    .str.replace("%", "", regex=False),
    errors="coerce"
)

# STEP 8: Remove rows missing essential information
essential_columns = [
    "product_id",
    "product_name",
    "discounted_price",
    "rating",
    "about_product",
    "review_content"
]

df = df.dropna(subset=essential_columns)

# Remove blank or placeholder text
for col in [
    "product_name",
    "about_product",
    "review_content"
]:
    text_values = df[col].astype(str).str.strip()
    df = df[
        text_values.ne("") & text_values.ne("-")
    ]

# STEP 9: Save the cleaned dataset
output_file = data_folder / "smartphones_clean.csv"

df.to_csv(output_file, index=False)

# STEP 10: Display the results
print("\nCleaning completed!")
print("Products remaining:", len(df))
print("Output saved to:", output_file)
print("\nFirst five products:")
print(df[["product_name", "discounted_price", "rating"]].head())
