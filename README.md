# PDF Q&A Agent

A lightweight Streamlit app that lets you upload PDFs, index their content into Pinecone using sentence-transformers embeddings, and ask natural-language questions answered by an OpenAI chat model (strictly using the retrieved PDF context).

##

Table of contents
1. Clone the repository
2. Tech Stack
3. Features
4. Prerequisites
5. Installation
6. Environment variables
7. Running the app
8. How to use
9. Implementation details and notes
10. Security & privacy
11. Troubleshooting & common issues
12. Extending the app
13. Contributing
14. License

---

1. Clone the repository

First, clone the repository to your local machine. 

Using HTTPS:
```
git clone https://github.com/Bhavesh-Mankar51/Document-AI-Project.git
```

Then change into the project directory:
```
cd <repo>
```

If you already cloned the repo, skip this step.

---

2. Tech Stack

- Language: Python 3.8+
- Web UI: Streamlit
- Embeddings: sentence-transformers (model: `all-MiniLM-L6-v2`)
- Vector DB: Pinecone (Serverless index used in code)
- LLM: OpenAI (e.g., `gpt-3.5-turbo`)
- PDF parsing: PyPDF2 (no OCR)
- Text splitting: langchain-text-splitters (RecursiveCharacterTextSplitter)
- Environment management: python-dotenv
- Optional / dev: git, virtualenv/venv

---

3. Features

- Upload one or more PDF files via Streamlit UI
- Extract text from PDFs (no OCR — PDFs with only images will be treated as empty)
- Split long text into chunks with overlap for better retrieval
- Create sentence embeddings (model: `all-MiniLM-L6-v2`) and store them in a Pinecone index
- Query Pinecone for relevant chunks and generate answers with OpenAI (e.g. `gpt-3.5-turbo`)
- Shows short excerpts as sources and the full generated answer

---

4. Prerequisites

- Python 3.8+
- Pinecone account and API key (Serverless index is used in the code)
- OpenAI account and API key
- Basic familiarity with terminals and virtual environments
- `git` installed (for cloning)

---

5. Installation

1. After cloning, create and activate a virtual environment (recommended):

   - macOS / Linux:
     ```
     python -m venv .venv
     source .venv/bin/activate
     ```

   - Windows (PowerShell):
     ```
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```

2. Install dependencies. If the repository already contains a `requirements.txt`, install from it:

   ```
   pip install -r requirements.txt
   ```

  

   Note: package names can vary between Pinecone SDK releases; match these to the Pinecone SDK you intend to use. The app uses `from pinecone import Pinecone, ServerlessSpec` (serverless API). If your Pinecone SDK differs, adjust imports and usage accordingly.

---

6. Environment variables

Create a `.env` file in the project root (do NOT commit this file). Example:

```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=pdf-qna-index
```

- `OPENAI_API_KEY`: Key for OpenAI API.
- `PINECONE_API_KEY`: Key for Pinecone.
- `PINECONE_INDEX_NAME`: Name of the index to create/use in Pinecone. The app will attempt to create the index if it doesn't exist (dimension 384, cosine).

The app uses `ServerlessSpec(cloud="aws", region="us-east-1")` when creating an index; if your Pinecone setup or preferred region differs, adjust this in the code.

---

7. Running the app

Run with Streamlit (assuming the main file is `app.py`; replace with your filename if different):

```
streamlit run app.py
```

Open the local URL Streamlit prints to your terminal (usually http://localhost:8501).

---

8. How to use

1. Open the Streamlit app in your browser.
2. Use the sidebar to upload one or more PDF files (accepted type: `.pdf`).
3. For each uploaded PDF, click the "Index" button to extract, chunk, embed, and store vectors in Pinecone. The app will indicate when a file is "Ready".
4. Type a natural-language question in the main input box and click "Search".
5. The app will retrieve the top-k chunks (default k=3) and send them as context to OpenAI to produce an answer strictly based on the retrieved context.
6. Expand "View Sources" to see the source excerpts and filenames.

Important behaviors:
- Uploaded files are tracked in `st.session_state.indexed_files` to avoid re-indexing the same file within the running session.
- If a PDF has only images (no embedded text), the app will raise a clear error: "PDF appears to be empty or contains only images (OCR not supported)."

---

9. Implementation details and notes

- Embedding model: `sentence-transformers` model `all-MiniLM-L6-v2` produces 384-dimensional embeddings.
- Text splitting: `RecursiveCharacterTextSplitter` with defaults (chunk_size=700, chunk_overlap=70) — tuned for small context windows and retrieval accuracy.
- Pinecone index: created with dimension 384 and `metric="cosine"`. The code uses serverless spec with `region="us-east-1"`. Change this if you use a different region or non-serverless index.
- OpenAI usage: the code uses `openai_client.chat.completions.create` with `gpt-3.5-turbo` and temperature 0 to prefer deterministic answers. Adjust model/params as you wish.
- The app currently sends a single prompt containing the entire concatenated retrieved context. If the context is long you may want to further refine prompting or use streaming/completions settings.

---

10. Security & privacy

- Do NOT commit your `.env`, API keys, or any secrets to version control.
- Uploaded PDFs and their text/chunks are stored only in memory/session state and in your configured Pinecone index. If Pinecone is a managed cloud vector DB, treat it like any other data storage — avoid uploading sensitive/private documents to third-party services if you don't want them stored outside your control.
- Consider adding user authentication or access controls when deploying publicly.

---

11. Troubleshooting & common issues

- "Missing Environment Variables. Check your .env file.": Make sure `.env` exists and contains the required keys. Restart the app after updating `.env`.
- PDF text extraction returns empty: The PDF likely contains images only (scanned document). The app does not perform OCR. To support scanned PDFs, add an OCR step (e.g., Tesseract + pytesseract) before extracting text.
- Pinecone index creation or upsert failures: Make sure your Pinecone key is valid and your account supports the index type used (serverless). Check Pinecone dashboard for errors or quotas.
- Long first run times: Downloading the sentence-transformers model (~10s–100s depending on connection) and initial API calls may take time. Subsequent runs are faster.
- Rate limits / API errors from OpenAI or Pinecone: Check rate limits and error messages. Consider exponential backoff and retries for robustness.

---

12. Extending the app

- Add OCR for scanned PDFs (pytesseract + Tesseract).
- Add metadata extraction (author, page numbers, headings).
- Support more sophisticated prompt engineering (chain-of-thought, citations, answer provenance).
- Add a "re-index" or "delete index" option in the UI to manage Pinecone contents.

---

13. Contributing

If you'd like to improve the repo:
- Open issues describing features or bugs.
- Add tests and CI where appropriate.
- Keep secrets out of the repo.

---

14. License

Add your chosen license (e.g., MIT) to the repository if you plan to publish it.