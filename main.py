# main.py

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv

# --- Import your refactored logic modules directly ---
# Ensure these files (book_recommender_logic.py, sentiment_analysis_logic.py,
# vector_search_logic.py) are in the SAME directory as main.py
from book_recommender_logic import initialize_recommender # Assuming you want to run this init
from vector_search_logic import initialize_vector_search, retrieve_semantic_recommendation
from sentiment_analysis_logic import initialize_sentiment_analyzer, process_book_description_for_emotions # Added process_book_description_for_emotions for potential future use

# --- Initialize FastAPI ---
app = FastAPI()

# --- Load Environment Variables ---
# It's good practice to load environment variables explicitly in your app
load_dotenv()

# --- CORS setup (so frontend can call this backend) ---
# In production, replace "*" with your specific frontend URL(s)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Startup Event to Initialize Models ---
# This decorator ensures that the `startup_event` function runs only once
# when the FastAPI application starts, loading all models and data.
@app.on_event("startup")
async def startup_event():
    """
    Initializes all necessary AI models and data when the FastAPI application starts.
    This prevents reloading models on every request and ensures they are ready.
    """
    print("--- Initializing all AI models and data ---")
    try:
        initialize_recommender()
        initialize_sentiment_analyzer()
        initialize_vector_search()
        print("--- All models and data initialized successfully ---")
    except Exception as e:
        print(f"--- ERROR during model initialization: {e} ---")
        # You might want to log this error more robustly or exit if critical
        # For now, the app will start but might fail on requests.

# --- Input schema for the recommendation endpoint ---
class UserQuery(BaseModel):
    description: str
    mood: str
    genre: str

# --- API Endpoints ---

@app.get("/")
async def read_root():
    """
    Root endpoint for the API.
    """
    return {"message": "Welcome to the Book Recommender API! Visit /docs for interactive API documentation."}

@app.post("/recommend")
def recommend_books(user_query: UserQuery):
    """
    Provides semantic book recommendations based on a user's description, mood, and genre.
    """
    if not user_query.description and not user_query.mood and not user_query.genre:
        raise HTTPException(status_code=400, detail="At least one of description, mood, or genre must be provided.")

    # Construct a comprehensive query text from user inputs
    query_parts = []
    if user_query.description:
        query_parts.append(user_query.description)
    if user_query.mood:
        query_parts.append(f"Mood: {user_query.mood}")
    if user_query.genre:
        query_parts.append(f"Genre: {user_query.genre}")

    query_text = ". ".join(query_parts) + "." # Combine parts into a single query string

    print(f"Received recommendation query: '{query_text}'")

    # Retrieve recommendations using the vector search logic
    # top_k=5 is a good default, you can make this configurable via UserQuery model if needed
    recommended_df = retrieve_semantic_recommendation(query_text, top_k=5)

    if recommended_df.empty:
        print("No recommendations found for the query.")
        return {"recommendations": []}

    # Format output for frontend consumption
    recommendations_list = []
    for _, row in recommended_df.iterrows():
        # Use .get() with a default value to safely access columns that might be missing
        recommendations_list.append({
            "isbn13": row.get("isbn13", "N/A"),
            "title": row.get("title_and_subtitle", row.get("title", "Unknown Title")), # Prefer combined title
            "author": row.get("authors", "Unknown Author"),
            "category": row.get("simple_categories", "Unknown Category"),
            "description": row.get("description", "No description available."),
            "average_rating": row.get("average_rating", None),
            "published_year": row.get("published_year", None),
            "image_url": row.get("image_url", None) # Include image URL if available in your data
        })

    print(f"Returning {len(recommendations_list)} recommendations.")
    return {"recommendations": recommendations_list}

if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 10000))  # fallback to 10000 for local dev
    uvicorn.run(app, host="0.0.0.0", port=port)



# You can add more endpoints here if needed, e.g., for direct sentiment analysis
# @app.post("/analyze-text")
# def analyze_text(request: SentimentAnalysisRequest):
#     """
#     Analyzes the sentiment of a given text using the sentiment analysis logic.
#     """
#     if not request.text:
#         raise HTTPException(status_code=400, detail="Text cannot be empty.")
#     
#     emotion_scores = process_book_description_for_emotions(request.text)
#     
#     if not emotion_scores:
#         raise HTTPException(status_code=500, detail="Could not analyze sentiment.")
#         
#     return {"text": request.text, "emotion_scores": emotion_scores}
