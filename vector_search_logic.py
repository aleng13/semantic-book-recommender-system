# vector_search_logic.py

import pandas as pd
import numpy as np
import os
import re
from dotenv import load_dotenv
from tqdm import tqdm # For progress bar

# LangChain components for text processing and vector database
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings # Updated import for HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma # Updated import for Chroma

# Hugging Face Transformers pipeline for zero-shot classification
from transformers import pipeline

# --- Global Variables for Loaded Components ---
# These variables will store the loaded data, embedding model, and vector database
# to avoid reloading them multiple times when the API is running.
CHROMA_DB = None
BOOKS_DF_FULL = None # Will store the full processed books DataFrame
ZERO_SHOT_CLASSIFIER_PIPE = None

# --- Configuration Constants ---
BOOKS_CLEANED_CSV_FILENAME = "books_cleaned.csv"
TAGGED_DESCRIPTION_TXT_FILENAME = "tagged_description.txt"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ZERO_SHOT_CLASSIFICATION_MODEL = "facebook/bart-large-mnli"

# Define a mapping for book categories for simplification
# This dict should cover all original 'categories' you want to simplify.
# If an original category is not in this map, it will result in NaN,
# which can then be handled by zero-shot classification.
CATEGORY_MAPPING = {
    "Fiction": "Fiction",
    "Juvenile Fiction": "Children's Fiction",
    "Biography & Autobiography": "Nonfiction",
    "History": "Nonfiction",
    "Literary Criticism": "Nonfiction",
    "Philosophy": "Nonfiction",
    "Religion": "Nonfiction",
    "Comics & Graphic Novels": "Fiction",
    "Drama": "Fiction",
    "Juvenile Nonfiction": "Children's Nonfiction",
    "Science": "Nonfiction",
    "Poetry": "Fiction",
    # Add more mappings if you have other common categories that need simplification
    # e.g., "Art": "Nonfiction", "Cooking": "Nonfiction" etc.
}

# Categories for zero-shot classification (for Fiction/Nonfiction distinction)
FICTION_NONFICTION_CATEGORIES = ["Fiction", "Nonfiction"]

# --- Data Preparation Functions ---

def prepare_data_for_embedding(books_df: pd.DataFrame,
                               output_filepath: str = TAGGED_DESCRIPTION_TXT_FILENAME):
    """
    Prepares the 'tagged_description' column of the DataFrame and saves it
    to a text file, one description per line. This file is then used by TextLoader.

    Args:
        books_df (pd.DataFrame): The DataFrame containing the 'tagged_description' column.
        output_filepath (str): The path to save the text file.
    """
    if "tagged_description" not in books_df.columns:
        raise ValueError("DataFrame must contain a 'tagged_description' column.")

    print(f"Exporting 'tagged_description' to {output_filepath}...")
    # Ensure all descriptions are strings and handle potential NaNs
    books_df["tagged_description"].astype(str).to_csv(
        output_filepath, index=False, header=False, encoding="utf-8"
    )
    print("Export complete.")

def load_documents_from_text_file(filepath: str) -> list:
    """
    Loads text documents from a specified file, with each line treated as a separate document.

    Args:
        filepath (str): The path to the text file.

    Returns:
        list: A list of LangChain Document objects.
    """
    print(f"Loading documents from {filepath}...")
    try:
        raw_documents = TextLoader(filepath, encoding="utf-8").load()
        # Using CharacterTextSplitter with chunk_size=0 and chunk_overlap=0
        # effectively keeps each line as a distinct document.
        text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
        documents = text_splitter.split_documents(raw_documents)
        print(f"Loaded {len(documents)} documents.")
        return documents
    except FileNotFoundError:
        print(f"Error: Document file not found at {filepath}")
        return []
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []

# --- Embedding and Vector DB Functions ---

def create_and_persist_chroma_db(documents: list,
                                 embedding_model_name: str = EMBEDDING_MODEL_NAME,
                                 persist_directory: str = "./chroma_db") -> Chroma:
    """
    Creates or loads a Chroma vector database from documents and an embedding model.
    The database will be persisted to disk.

    Args:
        documents (list): A list of LangChain Document objects.
        embedding_model_name (str): The name of the Sentence-Transformer model for embeddings.
        persist_directory (str): The directory to store the Chroma DB.

    Returns:
        Chroma: An initialized Chroma vector database instance.
    """
    print(f"Initializing HuggingFaceEmbeddings model: {embedding_model_name}...")
    embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)
    print("Embedding model loaded.")

    if not documents:
        print("No documents provided to create Chroma DB. Returning None.")
        return None

    print(f"Creating/loading Chroma DB from documents (persisting to {persist_directory})...")
    # Chroma.from_documents will create the DB and persist it
    db = Chroma.from_documents(documents, embedding=embedding, persist_directory=persist_directory)
    print("Chroma DB created/loaded successfully.")
    return db

# --- Zero-Shot Classification Functions ---

def load_zero_shot_classifier(model_name: str = ZERO_SHOT_CLASSIFICATION_MODEL):
    """
    Loads the zero-shot classification pipeline from Hugging Face Transformers.
    It attempts to use the GPU if available, otherwise falls back to CPU.

    Args:
        model_name (str): The name of the pre-trained zero-shot classification model.

    Returns:
        transformers.pipeline: The loaded zero-shot classification pipeline, or None if loading fails.
    """
    global ZERO_SHOT_CLASSIFIER_PIPE
    if ZERO_SHOT_CLASSIFIER_PIPE is None:
        print(f"Loading zero-shot classification model: {model_name}...")
        try:
            import torch
            device = 0 if torch.cuda.is_available() else -1
            print(f"Using device: {'GPU' if device == 0 else 'CPU'}")
            ZERO_SHOT_CLASSIFIER_PIPE = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=device
            )
            print("Zero-shot classifier loaded successfully.")
        except Exception as e:
            print(f"Error loading zero-shot classifier model: {e}")
            ZERO_SHOT_CLASSIFIER_PIPE = None
    return ZERO_SHOT_CLASSIFIER_PIPE

def predict_zero_shot_category(sequence: str, candidate_labels: list) -> str:
    """
    Predicts the best category for a given text sequence using the zero-shot classifier.

    Args:
        sequence (str): The text to classify (e.g., a book description).
        candidate_labels (list): A list of possible categories (e.g., ["Fiction", "Nonfiction"]).

    Returns:
        str: The predicted category label, or "Unknown" if classification fails.
    """
    if ZERO_SHOT_CLASSIFIER_PIPE is None:
        print("Zero-shot classifier not loaded. Cannot predict categories.")
        return "Unknown"
    if not isinstance(sequence, str) or not sequence.strip():
        return "Unknown" # Handle empty or invalid descriptions

    try:
        predictions = ZERO_SHOT_CLASSIFIER_PIPE(sequence, candidate_labels)
        # The model returns a dictionary with 'labels' and 'scores'.
        # The first label in 'labels' list is the highest-scoring prediction.
        return predictions["labels"][0]
    except Exception as e:
        print(f"Error predicting category for sequence: '{sequence[:50]}...' - {e}")
        return "Unknown"

def enrich_with_zero_shot_categories(books_df: pd.DataFrame,
                                     description_col: str = "description",
                                     category_col: str = "categories",
                                     simple_category_col: str = "simple_categories",
                                     isbn_col: str = "isbn13",
                                     candidate_labels: list = FICTION_NONFICTION_CATEGORIES) -> pd.DataFrame:
    """
    Enriches the DataFrame by filling missing 'simple_categories' using zero-shot classification.

    Args:
        books_df (pd.DataFrame): The DataFrame to enrich.
        description_col (str): Name of the column containing book descriptions.
        category_col (str): Name of the original categories column.
        simple_category_col (str): Name of the simplified categories column.
        isbn_col (str): Name of the ISBN column.
        candidate_labels (list): List of labels for zero-shot classification (e.g., ["Fiction", "Nonfiction"]).

    Returns:
        pd.DataFrame: The DataFrame with 'simple_categories' filled in.
    """
    print("Applying simple category mapping...")
    # Apply initial mapping first
    books_df[simple_category_col] = books_df[category_col].map(CATEGORY_MAPPING)

    # Load zero-shot classifier if not already loaded
    if load_zero_shot_classifier() is None:
        print("Zero-shot classifier could not be loaded. Skipping missing category prediction.")
        return books_df # Return DataFrame as is if classifier fails

    missing_cats_df = books_df[books_df[simple_category_col].isna()].copy()
    print(f"Found {len(missing_cats_df)} books with missing '{simple_category_col}'. Attempting zero-shot classification...")

    predicted_cats_list = []
    isbns_list = []

    # Iterate through books with missing simple categories and predict
    for index, row in tqdm(missing_cats_df.iterrows(), total=len(missing_cats_df), desc="Zero-shot predicting categories"):
        sequence = row[description_col]
        # Ensure sequence is a string; replace NaN with empty string
        if pd.isna(sequence):
            sequence = ""
        predicted_cat = predict_zero_shot_category(str(sequence), candidate_labels) # Cast to str to be safe
        predicted_cats_list.append(predicted_cat)
        isbns_list.append(row[isbn_col])

    # Create a DataFrame for the predictions
    missing_predicted_df = pd.DataFrame({isbn_col: isbns_list, "predicted_simple_categories": predicted_cats_list})

    # Merge predictions back into the main DataFrame
    books_df = pd.merge(books_df, missing_predicted_df, on=isbn_col, how="left")

    # Update simple_categories where it was NaN with the predicted category
    books_df[simple_category_col] = np.where(
        books_df[simple_category_col].isna(),
        books_df["predicted_simple_categories"],
        books_df[simple_category_col]
    )

    # Drop the temporary prediction column
    books_df = books_df.drop(columns=["predicted_simple_categories"], errors='ignore')

    print("Zero-shot classification for missing categories complete.")
    return books_df

# --- Core Recommendation Retrieval Function ---

def retrieve_semantic_recommendation(query: str,
                                     top_k: int = 10,
                                     max_db_search_results: int = 50) -> pd.DataFrame:
    """
    Retrieves semantic book recommendations from the Chroma DB based on a query.

    Args:
        query (str): The user's query for book recommendations.
        top_k (int): The number of final recommendations to return.
        max_db_search_results (int): The maximum number of results to retrieve from the
                                     vector database before filtering/processing. This allows
                                     for more robust filtering if needed.

    Returns:
        pd.DataFrame: A DataFrame of recommended books, or an empty DataFrame if no
                      recommendations are found or the DB is not initialized.
    """
    global CHROMA_DB, BOOKS_DF_FULL

    if CHROMA_DB is None or BOOKS_DF_FULL is None:
        print("Chroma DB or full book DataFrame not initialized. Attempting to initialize now...")
        initialize_vector_search()
        if CHROMA_DB is None or BOOKS_DF_FULL is None:
            print("Failed to initialize vector search. Cannot provide recommendations.")
            return pd.DataFrame()

    print(f"Performing semantic search for query: '{query}'")
    try:
        # Perform similarity search in the vector database
        recs = CHROMA_DB.similarity_search(query, k=max_db_search_results)

        # Extract ISBNs from the page_content of the retrieved documents
        # The format is assumed to be "isbn13_number description_text"
        books_isbns = []
        for doc in recs:
            # Use regex to robustly extract digits from the beginning of the content
            isbn_match = re.match(r'^\d+', doc.page_content.strip('"'))
            if isbn_match:
                try:
                    books_isbns.append(int(isbn_match.group(0)))
                except ValueError:
                    # Handle cases where conversion to int fails
                    print(f"Warning: Could not convert ISBN '{isbn_match.group(0)}' to int.")
                    pass

        # Filter the original DataFrame to get the full book details for recommended ISBNs
        # Using .isin() for efficient lookup of multiple ISBNs
        recommended_books_df = BOOKS_DF_FULL[BOOKS_DF_FULL["isbn13"].isin(books_isbns)].copy()

        # Optional: Re-rank by similarity score if you can preserve it from Chroma.
        # For simplicity, we just take the head(top_k) based on the order returned
        # by Chroma, which is usually similarity-ranked.
        if len(recommended_books_df) > top_k:
            # If Chroma returned more than top_k, ensure we only take the actual top_k.
            # This might require a more complex re-ordering if Chroma's list isn't strictly ordered.
            # For now, we rely on Chroma's inherent sorting.
            # A more robust way would be to get distance/score from Chroma and sort.
            # Example: docs_with_score = CHROMA_DB.similarity_search_with_score(query, k=max_db_search_results)
            # then sort by score.
            pass

        print(f"Retrieved {len(recommended_books_df)} detailed recommendations.")
        return recommended_books_df.head(top_k)

    except Exception as e:
        print(f"An error occurred during semantic recommendation retrieval: {e}")
        return pd.DataFrame()


# --- Initialization Function ---

def initialize_vector_search(books_df_path: str = BOOKS_CLEANED_CSV_FILENAME):
    """
    Initializes the vector search system:
    1. Loads the pre-cleaned book data.
    2. Enriches categories using zero-shot classification if needed.
    3. Prepares data for embedding (saves to a temp file).
    4. Loads documents for Chroma.
    5. Creates/loads the Chroma vector database.
    This function should be called once when the application starts.
    """
    global BOOKS_DF_FULL, CHROMA_DB

    load_dotenv() # Load environment variables (e.g., for HuggingFace/OpenAI if used later)

    if BOOKS_DF_FULL is None:
        print("Initializing vector search: Loading and preparing book data...")
        try:
            # Assuming books_cleaned.csv is already preprocessed from book_recommender_logic.py
            books_df_temp = pd.read_csv(books_df_path)
            # Apply zero-shot classification for categories
            BOOKS_DF_FULL = enrich_with_zero_shot_categories(books_df_temp.copy())
            print(f"Loaded and prepared {len(BOOKS_DF_FULL)} books.")
        except FileNotFoundError:
            print(f"Error: {books_df_path} not found. Please ensure it exists.")
            return
        except Exception as e:
            print(f"Error loading or preparing book data: {e}")
            return

    # Ensure the tagged_description.txt is created/updated for TextLoader
    prepare_data_for_embedding(BOOKS_DF_FULL)

    if CHROMA_DB is None:
        print("Initializing vector search: Creating/loading Chroma DB...")
        documents = load_documents_from_text_file(TAGGED_DESCRIPTION_TXT_FILENAME)
        if documents:
            CHROMA_DB = create_and_persist_chroma_db(documents)
        else:
            print("No documents loaded, cannot initialize Chroma DB.")
            return

    if BOOKS_DF_FULL is not None and CHROMA_DB is not None:
        print("Vector Search System fully initialized!")
    else:
        print("Vector Search System initialization incomplete.")


# --- Main execution block for testing ---
if __name__ == "__main__":
    print("--- Running vector_search_logic.py in stand-alone test mode ---")

    # This part should mimic how your main app.py would use this module.
    # It assumes `books_cleaned.csv` is available (generated by `book_recommender_logic.py`).

    # 1. Initialize the vector search system (loads data, creates/loads DB, etc.)
    initialize_vector_search()

    # If initialization was successful, proceed with tests
    if CHROMA_DB is not None and BOOKS_DF_FULL is not None:
        print("\n--- Testing Zero-Shot Category Enrichment (Example) ---")
        # Find a book that originally had a missing simple_category and see its prediction
        if "simple_categories" in BOOKS_DF_FULL.columns and "categories" in BOOKS_DF_FULL.columns:
            # Look for a book where simple_categories was likely filled by ZSC
            # This is hard to guarantee without knowing your exact data, so this is illustrative
            sample_book = BOOKS_DF_FULL[
                (BOOKS_DF_FULL["categories"].isin(["Cooking", "Sports"])) # Example original categories not in simple_map
                | (BOOKS_DF_FULL["simple_categories"].isna()) # Check books that were originally NaN
            ].head(1)
            if not sample_book.empty:
                print(f"Example book (ISBN: {sample_book['isbn13'].iloc[0]}):")
                print(f"  Original category: {sample_book['categories'].iloc[0]}")
                print(f"  Predicted/Simplified category: {sample_book['simple_categories'].iloc[0]}")
            else:
                print("Could not find a clear example of zero-shot category enrichment in the test subset.")
        else:
            print("Category columns not found for zero-shot testing.")

        print("\n--- Testing Semantic Recommendation Retrieval ---")
        test_queries = [
            "A thrilling detective story set in London",
            "Books on ancient Roman history for students",
            "Romantic novels with a happy ending",
            "Science fiction about artificial intelligence"
        ]

        for query in test_queries:
            print(f"\n--- Recommendations for '{query}' (Top 3) ---")
            recommendations_df = retrieve_semantic_recommendation(query, top_k=3)

            if not recommendations_df.empty:
                for i, row in recommendations_df.iterrows():
                    title = row.get('title_and_subtitle', row.get('title', 'N/A'))
                    author = row.get('authors', 'N/A')
                    category = row.get('simple_categories', 'N/A')
                    print(f"{i+1}. Title: {title}, Author: {author}, Category: {category}")
            else:
                print("No recommendations found.")
    else:
        print("\n--- Vector search system initialization failed. Cannot run further tests. ---")

    print("\n--- End of vector_search_logic.py test mode ---")