# 📚 Semantic Book Recommender

A Python-based book recommendation system that suggests books based on semantic search, categories, and emotional tone. Built using **LangChain**, **Chroma**, **HuggingFace embeddings**, and **Gradio** for a web interface.

---

## Features

- **Semantic Search**: Search books based on a query or story idea.
- **Category Filter**: Narrow recommendations by book categories.
- **Emotional Tone Filter**: Recommend books according to moods like Happy, Sad, Suspenseful, Angry, Surprising.
- **Thumbnail & Description Preview**: See book cover and a short description.
- **Interactive Web Interface**: Powered by Gradio.

---

## Tech Stack

- Python 3.10+
- Pandas, Numpy
- LangChain (Text splitting, embeddings, vector DB)
- Chroma (Vector database)
- HuggingFace Sentence Transformers
- Gradio (UI)

---

## Setup Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/mamillapalli-charishma/Book_Recommender.git
    cd Book_Recommender
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv .venv
    .venv\Scripts\activate   # Windows
    source .venv/bin/activate # Mac/Linux
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Add your `.env` file (do **not** commit it to GitHub):
    ```
    HUGGINGFACE_API_KEY=<your_key_here>
    ```

5. Run the Gradio dashboard:
    ```bash
    python gradio_dashboard.py
    ```

---

## Usage

- Open the Gradio interface in your browser.
- Enter a description or idea for a book.
- Select category and emotional tone.
- Get a gallery of recommended books.

---
git add README.md
git commit -m "Add project README"
git push origin main --force

## Notes

- Make sure `.env` is ignored in `.gitignore` to avoid leaking API keys.
- Requires internet connection to fetch embeddings from HuggingFace.

---

## License

MIT License
