
from pathlib import Path
import sqlite3
import pandas as pd

# Project paths
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "smartphones_clean.csv"
DB_PATH = BASE_DIR / "shopping_assistant.db"


def setup_database():
    # Check that the cleaned dataset exists
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {CSV_PATH}\n"
            "Please check your data folder."
        )

    # Load the cleaned dataset
    df = pd.read_csv(CSV_PATH)

    if df.empty:
        raise ValueError("The cleaned dataset is empty.")

    # Ensure product IDs are treated as text
    if "product_id" in df.columns:
        df["product_id"] = df["product_id"].astype(str)

    # Create or connect to the SQLite database
    with sqlite3.connect(DB_PATH) as connection:
        # Import the dataset into a products table
        df.to_sql(
            "products",
            connection,
            if_exists="replace",
            index=False
        )

        # Verify the imported data
        cursor = connection.cursor()

        cursor.execute("SELECT COUNT(*) FROM products")
        product_count = cursor.fetchone()[0]

        print("\nDatabase created successfully!")
        print(f"Database location: {DB_PATH}")
        print(f"Products imported: {product_count}")

        # Display a sample of products
        cursor.execute("""
            SELECT product_name, discounted_price, rating
            FROM products
            LIMIT 5
        """)

        print("\nSample products:")
        for product_name, price, rating in cursor.fetchall():
            print(
                f"- {product_name} | "
                f"Price: {price} | Rating: {rating}"
            )


if __name__ == "__main__":
    setup_database()
