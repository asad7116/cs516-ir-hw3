# CS516 – Homework 3: Local IR System (BM25)

This repository contains a local information retrieval (IR) system implemented in Python.
The system indexes a collection of news articles and supports BM25-based ranked retrieval
with basic evaluation metrics.

## 1. Project structure

```text
IR_HW-3/
├─ data/
│  └─ Articles.csv          # dataset
├─ src/
│  └─ ir_system.py          # main IR system (BM25 + metrics)
├─ notebooks/
│  └─ ir_system_colab.ipynb # original Colab notebook
├─ requirements.txt
└─ README.md

2. Setup

    1. Create and activate a virtual environment (optional but recommended):

        python -m venv .venv
        # Windows
        .venv\Scripts\activate
        # Linux / macOS
        # source .venv/bin/activate


    2. Install dependencies:

        pip install -r requirements.txt


    3. Make sure the dataset is available at:

        data/Articles.csv


    Alternatively, you can specify a custom path with an environment variable:

        set IR_DATA_PATH=FULL\PATH\TO\Articles.csv        # Windows
        # export IR_DATA_PATH=/full/path/to/Articles.csv   # Linux/macOS

3. Running the system

    From the project root:

        python -m src.ir_system


    The first run will download required NLTK resources (punkt, punkt_tab,
    stopwords, wordnet) if they are missing.

    You will see:

    Loaded NNNN documents from: .../data/Articles.csv
    Enter query (empty line to exit):
    Query>


    Type a query, for example:

    Query> oil price cash


    The system prints the top-5 documents with BM25 scores, headings, and text snippets.

    Press Enter on an empty line to exit.

4. Evaluation

    The system supports standard IR metrics:

    Precision

    Recall

    Average Precision (AP)

    Reciprocal Rank (RR)

    Relevance judgments are defined as sets of document indices. Inside
    src/ir_system.py, in main(), you can specify them like:

    RELEVANCE = {
        "oil price cash": {444, 575},  # example indices from the dataset
        # add more query → relevant-docs mappings here
    }


    When a query matches a key in RELEVANCE, the metrics are computed for the
    top-k results.

5. Colab usage

    The original development was done in Google Colab (notebooks/ir_system_colab.ipynb).

    In Colab:

    import os
    os.environ["IR_DATA_PATH"] = "/content/Articles.csv"

    from src.ir_system import BM25IRSystem

    system = BM25IRSystem()
    result = system.search("oil price cash", top_k=5)
    result["results_df"].head()


    This uses the same core code as the local version, but reads the dataset
    from /content/Articles.csv.

6. AI usage

    Parts of the project structure, refactoring, and documentation were assisted by an AI coding assistant (ChatGPT). The final code experiments, and testing were done by me.
