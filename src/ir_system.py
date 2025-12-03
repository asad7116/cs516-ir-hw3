from __future__ import annotations

from pathlib import Path
import os
from typing import List, Set, Tuple, Dict, Optional

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from rank_bm25 import BM25Okapi


# -------------------------------------------------------------------
# Paths & config
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "Articles.csv"


def get_data_path() -> Path:
    """
    Resolve the Articles.csv path.

    Priority:
    1) IR_DATA_PATH environment variable
    2) ./data/Articles.csv (default)
    """
    env_path = os.getenv("IR_DATA_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return DEFAULT_DATA_PATH


# -------------------------------------------------------------------
# NLTK setup
# -------------------------------------------------------------------

def ensure_nltk_resources() -> None:
    """Download required NLTK resources if not already present."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab/english", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
    ]
    for key, name in resources:
        try:
            nltk.data.find(key)
        except LookupError:
            nltk.download(name)


# ensure NLTK data exists before creating stopwords / lemmatizer
ensure_nltk_resources()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


# -------------------------------------------------------------------
# Preprocessing
# -------------------------------------------------------------------

def preprocess_text(text: str) -> List[str]:
    """Lowercase, tokenize, remove stopwords & punctuation, lemmatize."""
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    tokens = word_tokenize(text)

    cleaned: List[str] = []
    for tok in tokens:
        if not tok.isalnum():
            continue
        if tok in stop_words:
            continue
        cleaned.append(lemmatizer.lemmatize(tok))

    return cleaned


# -------------------------------------------------------------------
# Evaluation metrics
# -------------------------------------------------------------------

def calculate_precision_recall(
    retrieved: List[int],
    relevant: Set[int],
) -> Tuple[float, float]:
    retrieved_set = set(retrieved)
    tp = len(retrieved_set & relevant)
    precision = tp / len(retrieved) if retrieved else 0.0
    recall = tp / len(relevant) if relevant else 0.0
    return precision, recall


def average_precision(
    retrieved: List[int],
    relevant: Set[int],
) -> float:
    if not relevant:
        return 0.0

    hits = 0
    s = 0.0
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            hits += 1
            s += hits / i
    return s / len(relevant)


def reciprocal_rank(
    retrieved: List[int],
    relevant: Set[int],
) -> float:
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


# -------------------------------------------------------------------
# BM25 IR system
# -------------------------------------------------------------------

class BM25IRSystem:
    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or get_data_path()
        if not self.data_path.exists():
            raise FileNotFoundError(f"Articles.csv not found at: {self.data_path}")

        # Adjust encoding if needed; your CSV was latin-1
        self.df = pd.read_csv(self.data_path, encoding="latin-1")

        # Expect columns "Heading" and "Article"
        if "Article" not in self.df.columns:
            raise ValueError("Expected column 'Article' in Articles.csv")
        if "Heading" not in self.df.columns:
            # If your file used different heading column, adjust here
            raise ValueError("Expected column 'Heading' in Articles.csv")

        # Build BM25 index
        self.tokenized_docs: List[List[str]] = [
            preprocess_text(t) for t in self.df["Article"]
        ]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def search(
        self,
        query: str,
        top_k: int = 5,
        relevant_docs: Optional[Set[int]] = None,
    ) -> Dict:
        """Run BM25 search and (optionally) compute metrics."""
        query_tokens = preprocess_text(query)
        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = np.argsort(scores)[::-1][:top_k]
        ranked_scores = scores[ranked_indices]
        ranked_indices_list = [int(i) for i in ranked_indices]

        metrics: Dict[str, float] = {}
        if relevant_docs is not None:
            p, r = calculate_precision_recall(ranked_indices_list, relevant_docs)
            ap = average_precision(ranked_indices_list, relevant_docs)
            rr = reciprocal_rank(ranked_indices_list, relevant_docs)
            metrics = {
                "precision": p,
                "recall": r,
                "average_precision": ap,
                "reciprocal_rank": rr,
            }

        results_df = self.df.iloc[ranked_indices][["Heading", "Article"]].copy()
        results_df["score"] = ranked_scores

        return {
            "query": query,
            "indices": ranked_indices_list,
            "results_df": results_df,
            "metrics": metrics,
        }


# -------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------

def main() -> None:
    system = BM25IRSystem()
    print(f"Loaded {len(system.df)} documents from: {system.data_path}")
    print("Enter query (empty line to exit):\n")

    while True:
        q = input("Query> ").strip()
        if not q:
            print("Exiting.")
            break

        # For now, no relevance judgments (empty set)
        relevant_docs: Set[int] = {444,654,575,117,440}  # Example relevant doc IDs for testing

        out = system.search(q, top_k=5, relevant_docs=relevant_docs)

        print("\nTop results:")
        for i, (idx, row) in enumerate(out["results_df"].iterrows(), start=1):
            print(f"{i}. [doc {idx}] score={row['score']:.4f}")
            print(f"   Heading: {row['Heading']}")
            print(f"   Snippet: {row['Article'][:200]}...\n")

        if out["metrics"]:
            m = out["metrics"]
            print("Metrics:")
            print(f"  Precision:         {m['precision']:.4f}")
            print(f"  Recall:            {m['recall']:.4f}")
            print(f"  Average Precision: {m['average_precision']:.4f}")
            print(f"  Reciprocal Rank:   {m['reciprocal_rank']:.4f}")
        print("-" * 60)


if __name__ == "__main__":
    main()
