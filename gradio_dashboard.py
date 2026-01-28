import pandas as pd
import numpy as np
from dotenv import load_dotenv

import gradio as gr

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ------------------------------------------------------------------
# ENV
# ------------------------------------------------------------------
load_dotenv()

# ------------------------------------------------------------------
# DATA
# ------------------------------------------------------------------
books = pd.read_csv("books_with_emotions.csv")

books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# ------------------------------------------------------------------
# DOCUMENTS → VECTOR STORE
# ------------------------------------------------------------------
raw_documents = TextLoader(
    "tagged_description.txt",
    encoding="utf-8"
).load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

documents = text_splitter.split_documents(raw_documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db_books = Chroma.from_documents(
    documents=documents,
    embedding=embeddings
)

# ------------------------------------------------------------------
# RECOMMENDATION LOGIC
# ------------------------------------------------------------------
def retrieve_semantic_recommendations(
    query: str,
    category: str = "All",
    tone: str = "All",
    initial_top_k: int = 50,
    final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)

    # SAFE ISBN extraction
    books_list = []
    for rec in recs:
        first_token = rec.page_content.strip('"').split()[0]
        if first_token.isdigit():
            books_list.append(int(first_token))

    book_recs = books[books["isbn13"].isin(books_list)]

    if category != "All":
        book_recs = book_recs[
            book_recs["simple_categories"] == category
        ]

    if tone == "Happy":
        book_recs = book_recs.sort_values("joy", ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values("surprise", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values("anger", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values("fear", ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values("sadness", ascending=False)

    return book_recs.head(final_top_k)

# ------------------------------------------------------------------
# GRADIO WRAPPER
# ------------------------------------------------------------------
def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        desc = " ".join(row["description"].split()[:30]) + "..."

        authors = row["authors"].split(";")
        if len(authors) == 1:
            author_str = authors[0]
        elif len(authors) == 2:
            author_str = f"{authors[0]} and {authors[1]}"
        else:
            author_str = f"{', '.join(authors[:-1])}, and {authors[-1]}"

        caption = f"{row['title']} by {author_str}: {desc}"
        results.append((row["large_thumbnail"], caption))

    return results

# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks() as dashboard:
    gr.Markdown("# 📚 Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(
            label="Describe a book",
            placeholder="e.g. A story about forgiveness and redemption"
        )
        category_dropdown = gr.Dropdown(
            categories, value="All", label="Category"
        )
        tone_dropdown = gr.Dropdown(
            tones, value="All", label="Emotional tone"
        )
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(columns=8, rows=2)

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )

# ------------------------------------------------------------------
if __name__ == "__main__":
    dashboard.launch(theme=gr.themes.Glass())