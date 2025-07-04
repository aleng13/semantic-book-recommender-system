import pandas as pd
import numpy as np
import random
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

# Load environment variables (like OpenAI API Key)
load_dotenv()

# Load the dataset
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(), "cover-not-found.jpg", books["large_thumbnail"]
)

# Load text documents and create vector database
raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_books = Chroma.from_documents(documents, embedding_model)

# Book quotes to randomly display on UI load
book_quotes = [
    "\"It is our choices, Harry, that show what we truly are.\" – J.K. Rowling",
    "\"So many books, so little time.\" – Frank Zappa",
    "\"A reader lives a thousand lives before he dies.\" – George R.R. Martin",
    "\"Until I feared I would lose it, I never loved to read.\" – Harper Lee",
    "\"There is no friend as loyal as a book.\" – Ernest Hemingway"
]

def get_random_quote():
    return random.choice(book_quotes)

def retrieve_semantic_recommendations(query, category=None, tone=None, initial_top_k=50, final_top_k=16):
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs

def recommend_books(query, category, tone):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"**{row['title']}** by _{authors_str}_\n\n{truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

# UI options
categories = ["All"] + sorted(books["simple_categories"].dropna().unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Build the Gradio dashboard
with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# \\U0001F4DA Semantic Book Recommender")
    gr.Markdown(f"_\\\"{get_random_quote()}\\\"_")

    with gr.Row():
        user_query = gr.Textbox(label="Enter your book description or theme:",
                                placeholder="e.g., A tale of love and redemption")
        category_dropdown = gr.Dropdown(choices=categories, label="Choose category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select emotional tone:", value="All")
        submit_button = gr.Button("\U0001F50D Recommend")

    gr.Markdown("---")
    gr.Markdown("## \\U0001F4C4 Your Recommendations")
    output = gr.Gallery(label="Books You Might Like", columns=4, rows=2, height="auto")

    submit_button.click(fn=recommend_books, inputs=[user_query, category_dropdown, tone_dropdown], outputs=output)

if __name__ == "__main__":
    dashboard.launch()
