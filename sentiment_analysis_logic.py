# sentiment_analysis_logic.py

import pandas as pd
import numpy as np
from transformers import pipeline
from tqdm import tqdm # For progress bar when processing many descriptions
import os # For path manipulation

# --- Global Variable for Loaded Classifier ---
# This will store the loaded Hugging Face pipeline classifier to avoid
# reloading it multiple times when the API is running.
EMOTION_CLASSIFIER = None

# --- Configuration Constants ---
# Define constants for model names and emotion labels.
EMOTION_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
# The order of these labels is important as the model's output scores
# are often fixed in a specific order. Ensure this matches the model's output order.
# Based on your previous code: ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

# --- Model Loading Function ---

def load_emotion_classifier(model_name: str = EMOTION_MODEL_NAME):
    """
    Loads the emotion classification model from Hugging Face Transformers.
    It attempts to use the GPU if available, otherwise falls back to CPU.

    Args:
        model_name (str): The name of the pre-trained emotion classification model.

    Returns:
        transformers.pipeline: The loaded text classification pipeline, or None if loading fails.
    """
    global EMOTION_CLASSIFIER
    if EMOTION_CLASSIFIER is None:
        print(f"Loading emotion classification model: {model_name}...")
        try:
            # Try to use GPU (device=0) if CUDA is available, otherwise use CPU (device=-1)
            # This is a common pattern for optimizing performance.
            import torch
            device = 0 if torch.cuda.is_available() else -1
            print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

            EMOTION_CLASSIFIER = pipeline(
                "text-classification",
                model=model_name,
                top_k=None,  # Get scores for all classes
                device=device # -1 for CPU, 0 for first GPU
            )
            print("Emotion classifier loaded successfully.")
        except Exception as e:
            print(f"Error loading emotion classifier model: {e}")
            EMOTION_CLASSIFIER = None
    return EMOTION_CLASSIFIER

# --- Helper Functions for Emotion Calculation ---

def calculate_max_emotion_scores(predictions: list) -> dict:
    """
    Calculates the maximum score for each emotion across a list of sentence predictions
    for a single book's description.

    Args:
        predictions (list): A list of lists, where each inner list contains dictionaries
                            of {'label': 'emotion', 'score': float} for a sentence.
                            Example: [[{'label': 'joy', 'score': 0.9}, ...], ...]

    Returns:
        dict: A dictionary where keys are emotion labels and values are the
              maximum score observed for that emotion across all sentences.
    """
    if not predictions:
        return {label: 0.0 for label in EMOTION_LABELS} # Return zeros if no predictions

    # Dictionary to hold all scores collected for each emotion across all sentences
    per_emotion_scores = {label: [] for label in EMOTION_LABELS}

    for sentence_prediction in predictions:
        # Sort predictions per sentence alphabetically by label to ensure consistent indexing
        # This is crucial because the EMOTION_LABELS list is also sorted alphabetically
        # (e.g., "anger", "disgust", "fear", ...) to match model output conventions.
        sorted_sentence_prediction = sorted(sentence_prediction, key=lambda x: x["label"])

        for index, label in enumerate(EMOTION_LABELS):
            # Append the score for the current label from the sorted prediction
            per_emotion_scores[label].append(sorted_sentence_prediction[index]["score"])

    # For each emotion, get the maximum score observed across all sentences for the book
    # If an emotion had no scores (e.g., empty input), np.max will throw error; handle by providing 0.0
    max_scores = {
        label: np.max(scores) if scores else 0.0
        for label, scores in per_emotion_scores.items()
    }
    return max_scores

def process_book_description_for_emotions(description: str) -> dict:
    """
    Analyzes a single book's description for emotions by splitting it into sentences
    and taking the max score for each emotion across all sentences.

    Args:
        description (str): The text description of a book.

    Returns:
        dict: A dictionary containing the maximum emotion scores for the book.
              Returns zeros for all emotions if description is invalid or classifier not loaded.
    """
    if EMOTION_CLASSIFIER is None:
        print("Emotion classifier not loaded. Cannot process description.")
        return {label: 0.0 for label in EMOTION_LABELS}

    if not isinstance(description, str) or not description.strip():
        # Handle cases where description is not a string or is empty
        return {label: 0.0 for label in EMOTION_LABELS}

    # Split description into sentences. Using '.' as a simple delimiter.
    # Consider using more robust sentence tokenizers (e.g., NLTK's sent_tokenize)
    # for production, but '.' works for initial implementation.
    sentences = [s.strip() for s in description.split(".") if s.strip()]

    if not sentences:
        return {label: 0.0 for label in EMOTION_LABELS} # No valid sentences found

    try:
        # Get predictions for all sentences in the description
        predictions = EMOTION_CLASSIFIER(sentences)
        # Calculate max scores across all sentences for the book
        max_scores = calculate_max_emotion_scores(predictions)
        return max_scores
    except Exception as e:
        print(f"Error processing description for emotions: {e}")
        return {label: 0.0 for label in EMOTION_LABELS}

# --- Main Function for Batch Processing ---

def add_emotion_scores_to_dataframe(df: pd.DataFrame,
                                    description_column: str = "description",
                                    isbn_column: str = "isbn13") -> pd.DataFrame:
    """
    Processes all book descriptions in a DataFrame to add emotion scores as new columns.

    Args:
        df (pd.DataFrame): The DataFrame containing book data.
        description_column (str): The name of the column containing book descriptions.
        isbn_column (str): The name of the column containing book ISBNs (for tracking).

    Returns:
        pd.DataFrame: The original DataFrame with new columns for each emotion's score.
                      Returns the original DataFrame unchanged if classifier fails to load.
    """
    # Ensure classifier is loaded before processing
    if load_emotion_classifier() is None:
        print("Emotion classifier not available. Skipping emotion analysis.")
        return df

    print(f"Starting emotion analysis for {len(df)} books...")

    # Initialize lists to store results
    isbns = []
    emotion_data = {label: [] for label in EMOTION_LABELS}

    # Iterate through each book using tqdm for a progress bar
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing emotions"):
        isbn = row[isbn_column]
        description = row[description_column]

        # Process the description and get max emotion scores
        max_scores = process_book_description_for_emotions(description)

        # Append results
        isbns.append(isbn)
        for label in EMOTION_LABELS:
            emotion_data[label].append(max_scores[label])

    # Create a DataFrame from the collected emotion scores
    emotions_df = pd.DataFrame(emotion_data)
    emotions_df[isbn_column] = isbns

    print("Emotion analysis complete. Merging results with original DataFrame.")
    # Merge the emotion scores back to the original DataFrame
    # Using 'left' merge to keep all original books even if some couldn't be processed
    merged_df = pd.merge(df, emotions_df, on=isbn_column, how="left")

    # Fill NaN values for emotion columns that might occur if some descriptions
    # failed to be processed and thus weren't in emotions_df for merging.
    for label in EMOTION_LABELS:
        if label not in merged_df.columns:
            merged_df[label] = 0.0 # Add column if somehow missed
        merged_df[label] = merged_df[label].fillna(0.0) # Fill any NaNs with 0.0

    print("Emotion scores successfully added to DataFrame.")
    return merged_df

# --- Initialization Function ---

def initialize_sentiment_analyzer():
    """
    Initializes the sentiment analyzer by loading the emotion classification model.
    This function should be called once when the application starts.
    """
    print("Initializing sentiment analyzer...")
    load_emotion_classifier()
    if EMOTION_CLASSIFIER is not None:
        print("Sentiment analyzer fully initialized!")
    else:
        print("Sentiment analyzer initialization failed.")

# --- Main execution block for testing ---
if __name__ == "__main__":
    print("--- Running sentiment_analysis_logic.py in stand-alone test mode ---")

    # Ensure you have a 'books_with_categories.csv' in the same directory
    # or adjust the path.
    # For a real application, the main app.py would load this DataFrame.
    try:
        # Attempt to load a dummy or existing CSV for testing
        if os.path.exists("books_with_categories.csv"):
            books_test_df = pd.read_csv("books_with_categories.csv")
            print(f"Loaded {len(books_test_df)} books from books_with_categories.csv for testing.")
        else:
            print("books_with_categories.csv not found. Creating a dummy DataFrame for testing.")
            books_test_df = pd.DataFrame({
                "isbn13": [9780132128793, 9780132128809, 9780132128816, 9780132128823, 9780132128830],
                "description": [
                    "This is a heartwarming story about joy and new beginnings. It fills you with delight.",
                    "A terrifying horror novel that will evoke fear and disgust. Not for the faint of heart.",
                    "A neutral description of a technical manual. Very informative, no strong emotions.",
                    "The protagonist experiences profound sadness and anger after a betrayal.",
                    "A surprising twist in this adventure novel makes for an exciting read."
                ],
                "title": ["Joyful Journey", "Horror Night", "Tech Manual", "Betrayal", "Surprise Adventure"]
            })
            print("Dummy DataFrame created.")

        # Initialize the sentiment analyzer
        initialize_sentiment_analyzer()

        if EMOTION_CLASSIFIER:
            # Test individual description processing
            print("\n--- Testing individual description processing ---")
            sample_desc = books_test_df["description"].iloc[0]
            print(f"Sample description: {sample_desc[:100]}...")
            individual_scores = process_book_description_for_emotions(sample_desc)
            print("Individual Emotion Scores:")
            for label, score in individual_scores.items():
                print(f"  {label}: {score:.4f}")

            # Test batch processing for the entire DataFrame
            print("\n--- Testing batch processing for DataFrame ---")
            # Process only a small subset for quick testing if the DataFrame is large
            test_subset_df = books_test_df.head(50).copy() if len(books_test_df) > 50 else books_test_df.copy()
            processed_books_df = add_emotion_scores_to_dataframe(test_subset_df)

            print("\nProcessed DataFrame head with emotion scores:")
            print(processed_books_df[
                ["isbn13", "title"] + EMOTION_LABELS
            ].head())

            # Verify that emotion columns are present and filled
            print(f"\nColumns added: {EMOTION_LABELS}")
            print(f"Missing values in emotion columns after processing:\n{processed_books_df[EMOTION_LABELS].isna().sum()}")

            # Save the processed data to a new CSV (optional, for verification)
            # processed_books_df.to_csv("books_with_emotions_test.csv", index=False)
            # print("\nProcessed data saved to books_with_emotions_test.csv (if not a dummy).")

        else:
            print("\n--- Sentiment analyzer could not be initialized. Skipping tests. ---")

    except FileNotFoundError:
        print("Error: 'books_with_categories.csv' not found. Please ensure it's in the same directory for testing.")
    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")

    print("\n--- End of sentiment_analysis_logic.py test mode ---")