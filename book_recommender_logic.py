# book_recommender_logic.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import kagglehub
import os # To handle file paths

# --- Global Variables for Loaded Data and Model ---
# These variables will store the loaded data and model to avoid reloading them
# multiple times when the API is running.
BOOK_DF = None
MODEL = None
BOOK_EMBEDDINGS = None

# --- Configuration Constants ---
# Define constants for file paths and model names to make them easily configurable.
KAGGLE_DATASET_REF = "dylanjcastillo/7k-books-with-metadata"
BOOKS_CSV_FILENAME = "books.csv"
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
MIN_DESCRIPTION_WORDS = 25 # Minimum words for a meaningful description

# --- Data Loading and Preprocessing Functions ---

def download_and_load_data(dataset_ref: str = KAGGLE_DATASET_REF,
                           csv_filename: str = BOOKS_CSV_FILENAME) -> pd.DataFrame:
    """
    Downloads the specified Kaggle dataset and loads the books CSV into a DataFrame.

    Args:
        dataset_ref (str): The reference string for the Kaggle dataset.
                           e.g., "dylanjcastillo/7k-books-with-metadata"
        csv_filename (str): The name of the CSV file within the dataset.
                           e.g., "books.csv"

    Returns:
        pd.DataFrame: A DataFrame containing the book data.
    """
    print(f"Downloading Kaggle dataset: {dataset_ref}")
    try:
        path = kagglehub.dataset_download(dataset_ref)
        file_path = os.path.join(path, csv_filename)
        books_df = pd.read_csv(file_path)
        print(f"Successfully loaded {len(books_df)} books from {file_path}")
        return books_df
    except Exception as e:
        print(f"Error downloading or loading data: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

def plot_missing_values_heatmap(df: pd.DataFrame, title: str = "Missing Values Heatmap"):
    """
    Plots a heatmap showing missing values in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        title (str): The title for the plot.
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(df.isna().transpose(), cbar=False, cmap='viridis') # Using 'viridis' for better contrast
    plt.xlabel("Rows (Book Entries)")
    plt.ylabel("Columns (Features with Missing Values)")
    plt.title(title)
    plt.show()

def plot_correlation_heatmap(df: pd.DataFrame, columns_of_interest: list, title: str = "Correlation Heatmap"):
    """
    Calculates and plots a Spearman correlation heatmap for specified columns.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        columns_of_interest (list): A list of column names for correlation analysis.
        title (str): The title for the plot.
    """
    # Create auxiliary columns for correlation analysis if they don't exist
    if "missing_description" not in df.columns:
        df["missing_description"] = np.where(df["description"].isna(), 1, 0)
    if "age_of_book" not in df.columns and "published_year" in df.columns:
        df["age_of_book"] = 2024 - df["published_year"]

    correlation_matrix = df[columns_of_interest].corr(method="spearman")

    sns.set_theme(style="white")
    plt.figure(figsize=(8, 6))
    heatmap = sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar_kws={"label": "Spearman Correlation"}
    )
    heatmap.set_title(title)
    plt.show()

def preprocess_books_data(books_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and preprocesses the book data for recommendation.
    This includes handling missing values, creating derived features,
    and ensuring descriptions are meaningful.

    Args:
        books_df (pd.DataFrame): The raw DataFrame of books.

    Returns:
        pd.DataFrame: The cleaned and preprocessed DataFrame.
    """
    print("Starting data preprocessing...")

    # 1. Handle missing core fields
    initial_rows = len(books_df)
    books_cleaned = books_df[
        books_df["description"].notna() &
        books_df["num_pages"].notna() &
        books_df["average_rating"].notna() &
        books_df["published_year"].notna()
    ].copy()
    print(f"Removed {initial_rows - len(books_cleaned)} rows with missing critical information.")
    print(f"Remaining books after initial cleaning: {len(books_cleaned)}")

    # 2. Calculate words in description
    books_cleaned["words_in_description"] = books_cleaned["description"].str.split().str.len()

    # 3. Filter for meaningful descriptions (e.g., at least 25 words)
    books_filtered_description = books_cleaned[
        books_cleaned["words_in_description"] >= MIN_DESCRIPTION_WORDS
    ].copy()
    print(f"Filtered to {len(books_filtered_description)} books with >= {MIN_DESCRIPTION_WORDS} words in description.")

    # 4. Create 'title_and_subtitle'
    # Use .fillna('') to ensure no NaNs before joining, then replace empty strings with NaN if needed
    books_filtered_description["title_and_subtitle"] = np.where(
        books_filtered_description["subtitle"].isna(),
        books_filtered_description["title"],
        books_filtered_description["title"] + ":" + books_filtered_description["subtitle"]
    )
    # Ensure combined title isn't just "Title:nan" if subtitle was NaN but not handled by np.where
    books_filtered_description["title_and_subtitle"] = books_filtered_description["title_and_subtitle"].replace("nan", "", regex=True).str.strip(':')

    # 5. Create 'tagged_description' for uniqueness in embeddings
    # Concatenate isbn13 as string with description
    books_filtered_description["tagged_description"] = \
        books_filtered_description["isbn13"].astype(str) + " " + books_filtered_description["description"]

    print("Data preprocessing complete.")
    return books_filtered_description

# --- Model Loading and Embedding Generation ---

def load_sentence_transformer_model(model_name: str = SENTENCE_TRANSFORMER_MODEL) -> SentenceTransformer:
    """
    Loads a SentenceTransformer model.

    Args:
        model_name (str): The name of the pre-trained SentenceTransformer model.

    Returns:
        SentenceTransformer: The loaded model.
    """
    print(f"Loading SentenceTransformer model: {model_name}")
    try:
        model = SentenceTransformer(model_name)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading SentenceTransformer model: {e}")
        return None

def generate_book_embeddings(df: pd.DataFrame, model: SentenceTransformer) -> np.ndarray:
    """
    Generates embeddings for book descriptions using the loaded SentenceTransformer model.

    Args:
        df (pd.DataFrame): The DataFrame containing processed book data with 'tagged_description'.
        model (SentenceTransformer): The loaded SentenceTransformer model.

    Returns:
        np.ndarray: A NumPy array of embeddings.
    """
    if model is None:
        raise ValueError("SentenceTransformer model is not loaded. Cannot generate embeddings.")
    if "tagged_description" not in df.columns:
        raise ValueError("DataFrame must contain a 'tagged_description' column.")

    print("Generating book embeddings...")
    # Use .tolist() for better performance when passing to model.encode
    descriptions = df["tagged_description"].fillna("").tolist()
    embeddings = model.encode(descriptions, show_progress_bar=True)
    print("Book embeddings generated.")
    return embeddings

# --- Core Recommendation Logic ---

def get_recommendations(user_query: str, num_results: int = 10) -> list:
    """
    Generates book recommendations based on a user query using semantic similarity.

    Args:
        user_query (str): The text query from the user (e.g., "fantasy adventures").
        num_results (int): The number of top recommendations to return.

    Returns:
        list: A list of dictionaries, where each dictionary represents a recommended book
              and its similarity score.
              Returns an empty list if data/model not loaded or no results found.
    """
    global BOOK_DF, MODEL, BOOK_EMBEDDINGS

    # Ensure all necessary components are loaded
    if BOOK_DF is None or MODEL is None or BOOK_EMBEDDINGS is None:
        print("Required data, model, or embeddings not loaded. Attempting to load now...")
        initialize_recommender() # Call the initialization function

        if BOOK_DF is None or MODEL is None or BOOK_EMBEDDINGS is None:
            print("Failed to initialize recommender. Cannot provide recommendations.")
            return []

    print(f"Generating recommendations for query: '{user_query}'")

    try:
        # Encode the user query
        query_embedding = MODEL.encode([user_query])[0]

        # Calculate cosine similarity between query and all book embeddings
        similarities = cosine_similarity([query_embedding], BOOK_EMBEDDINGS)[0]

        # Get the indices of the top N most similar books
        # np.argsort returns indices that would sort an array. [::-1] reverses it for descending order.
        top_indices = np.argsort(similarities)[::-1][:num_results]

        # Retrieve the recommended books and their similarity scores
        recommended_books = []
        for i in top_indices:
            book_info = BOOK_DF.iloc[i].to_dict()
            book_info['similarity_score'] = similarities[i] # Add similarity score
            recommended_books.append(book_info)

        print(f"Found {len(recommended_books)} recommendations.")
        return recommended_books

    except Exception as e:
        print(f"An error occurred during recommendation generation: {e}")
        return []

# --- Initialization Function ---

def initialize_recommender():
    """
    Initializes the book recommender system by loading data, preprocessing,
    loading the model, and generating embeddings.
    This function should be called once when the application starts.
    """
    global BOOK_DF, MODEL, BOOK_EMBEDDINGS

    if BOOK_DF is None:
        print("Initializing recommender: Downloading and loading data...")
        raw_books_df = download_and_load_data()
        if not raw_books_df.empty:
            BOOK_DF = preprocess_books_data(raw_books_df)
        else:
            print("Initialization failed: Could not load raw book data.")
            return

    if MODEL is None:
        print("Initializing recommender: Loading SentenceTransformer model...")
        MODEL = load_sentence_transformer_model()
        if MODEL is None:
            print("Initialization failed: Could not load SentenceTransformer model.")
            return

    if BOOK_EMBEDDINGS is None and BOOK_DF is not None and MODEL is not None:
        print("Initializing recommender: Generating book embeddings...")
        try:
            BOOK_EMBEDDINGS = generate_book_embeddings(BOOK_DF, MODEL)
        except Exception as e:
            print(f"Initialization failed: Could not generate embeddings - {e}")
            BOOK_EMBEDDINGS = None # Reset to None if generation failed
            return

    if BOOK_DF is not None and MODEL is not None and BOOK_EMBEDDINGS is not None:
        print("Book Recommender System fully initialized!")
    else:
        print("Book Recommender System initialization incomplete.")


# --- Main execution block for testing ---
if __name__ == "__main__":
    print("--- Running book_recommender_logic.py in stand-alone test mode ---")

    # 1. Initialize the recommender (loads data, model, generates embeddings)
    initialize_recommender()

    # If initialization was successful, proceed with tests
    if BOOK_DF is not None and MODEL is not None and BOOK_EMBEDDINGS is not None:
        print("\n--- Testing Data Analysis Visualizations ---")
        # You can uncomment these if you want to see the plots during local testing
        # plot_missing_values_heatmap(BOOK_DF, title="Missing Values in Preprocessed Data")
        # plot_correlation_heatmap(
        #     BOOK_DF,
        #     columns_of_interest=["num_pages", "age_of_book", "missing_description", "average_rating"],
        #     title="Correlation in Preprocessed Data"
        # )

        print("\n--- Testing Recommendation Function ---")
        test_queries = [
            "science fiction adventure",
            "historical drama",
            "cookbooks for beginners",
            "poetry about nature",
            "biographies of famous scientists"
        ]

        for query in test_queries:
            print(f"\n--- Recommendations for '{query}' (Top 3) ---")
            recommendations = get_recommendations(query, num_results=3)

            if recommendations:
                for i, book in enumerate(recommendations):
                    title = book.get('title_and_subtitle', book.get('title', 'N/A'))
                    author = book.get('authors', 'N/A')
                    rating = book.get('average_rating', 'N/A')
                    similarity = book.get('similarity_score', 'N/A')
                    print(f"{i+1}. Title: {title}, Author: {author}, Rating: {rating:.2f}, Similarity: {similarity:.4f}")
            else:
                print("No recommendations found.")
    else:
        print("\n--- Recommender initialization failed. Cannot run further tests. ---")

    print("\n--- End of book_recommender_logic.py test mode ---")