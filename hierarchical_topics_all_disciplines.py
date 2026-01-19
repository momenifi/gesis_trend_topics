#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BERTopic analysis across all disciplines combined (no discipline split).

CSV format ($$ separator):
title$$abstract$$year$$source$$keyword$$class_name$$middle_group$$subject_area
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
import umap
import hdbscan


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
DATA_PATH = "abstracts.csv"
OUTPUT_DIR = "outputs_all_disciplines"

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

MIN_CLUSTER_SIZE = 10
N_COMPONENTS = 5
N_NEIGHBORS = 15
RANDOM_STATE = 42

# Lower bound for running topic modeling at all
MIN_DOCS_FOR_CLUSTER = 10

DEFAULT_CONFIG = {
    "data_path": DATA_PATH,
    "output_dir": OUTPUT_DIR,
    "embedding_model": EMBEDDING_MODEL,
    "min_cluster_size": MIN_CLUSTER_SIZE,
    "n_components": N_COMPONENTS,
    "n_neighbors": N_NEIGHBORS,
    "random_state": RANDOM_STATE,
    "min_docs_for_cluster": MIN_DOCS_FOR_CLUSTER,
}


def load_yaml_config(config_path: str) -> dict:
    """Load YAML configuration and ensure we get a mapping."""
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a YAML mapping.")

    return data


def apply_config_overrides(config: dict):
    """Apply configuration values to module-level defaults."""
    global DATA_PATH, OUTPUT_DIR, EMBEDDING_MODEL
    global MIN_CLUSTER_SIZE, N_COMPONENTS, N_NEIGHBORS, RANDOM_STATE
    global MIN_DOCS_FOR_CLUSTER

    DATA_PATH = config.get("data_path", DATA_PATH)
    OUTPUT_DIR = config.get("output_dir", OUTPUT_DIR)
    EMBEDDING_MODEL = config.get("embedding_model", EMBEDDING_MODEL)

    MIN_CLUSTER_SIZE = int(config.get("min_cluster_size", MIN_CLUSTER_SIZE))
    N_COMPONENTS = int(config.get("n_components", N_COMPONENTS))
    N_NEIGHBORS = int(config.get("n_neighbors", N_NEIGHBORS))
    RANDOM_STATE = int(config.get("random_state", RANDOM_STATE))
    MIN_DOCS_FOR_CLUSTER = int(config.get("min_docs_for_cluster", MIN_DOCS_FOR_CLUSTER))


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def simple_clean(text: str) -> str:
    """Basic text cleaning for abstracts."""
    if not isinstance(text, str):
        return ""
    text = text.strip()

    # Handle abstracts wrapped like {" ... "}
    if text.startswith('{"') and text.endswith('"}'):
        text = text[2:-2]

    text = text.replace("\\n", " ")
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_linear_trends(count_table: pd.DataFrame, top_n: int = 10):
    """Compute linear trends per topic from a year x topic table."""
    slopes = []
    x = np.arange(len(count_table.index), dtype=float)

    for topic in count_table.columns:
        y = count_table[topic].values.astype(float)
        if len(y) < 2 or np.all(y == y[0]):
            m = 0.0
        else:
            m, _ = np.polyfit(x, y, 1)
        slopes.append((topic, m))

    slopes_sorted = sorted(slopes, key=lambda t: t[1], reverse=True)
    top_rising = slopes_sorted[:top_n]
    top_falling = slopes_sorted[-top_n:]
    return top_rising, top_falling


def build_vectorizer(stopwords, n_docs: int) -> CountVectorizer:
    """Create a CountVectorizer with dynamic min_df and robust settings."""
    if n_docs >= 200:
        min_df = 5
    elif n_docs >= 50:
        min_df = 3
    elif n_docs >= 20:
        min_df = 2
    else:
        min_df = 1

    vectorizer = CountVectorizer(
        stop_words=stopwords,
        ngram_range=(1, 3),
        min_df=min_df,
        max_df=0.95,
    )
    return vectorizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the BERTopic pipeline across all disciplines together."
    )
    parser.add_argument(
        "-c",
        "--config",
        help=(
            "Path to a YAML configuration file overriding defaults such as "
            "data_path or embedding_model. Example: config.example.yaml"
        ),
    )
    return parser.parse_args()


def run_bertopic_all(df: pd.DataFrame):
    """Run the full BERTopic pipeline on the complete dataset."""
    if df.empty:
        print("No documents available; skipping analysis.")
        return

    if len(df) < MIN_DOCS_FOR_CLUSTER:
        print(f"Fewer than {MIN_DOCS_FOR_CLUSTER} documents; topic modeling is skipped.")
        return

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    df = df.dropna(subset=["abstract", "year"]).copy()
    df["year_raw"] = df["year"]
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    df = df.sort_values("year").reset_index(drop=True)

    df["clean_text"] = df["abstract"].astype(str).apply(simple_clean)
    docs = df["clean_text"].tolist()
    years = df["year"].tolist()
    n_docs = len(df)

    print(f"Years in dataset: {df['year'].min()} to {df['year'].max()} (n={n_docs})")
    print(f"Using EMBEDDING_MODEL in BERTopic: {EMBEDDING_MODEL}")

    # BERTopic configuration
    umap_model = umap.UMAP(
        n_neighbors=N_NEIGHBORS,
        n_components=N_COMPONENTS,
        min_dist=0.0,
        metric="cosine",
        random_state=RANDOM_STATE,
    )

    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    extra_stopwords = {
        "research", "researchers", "study", "studies", "analysis", "analyses",
        "survey", "surveys", "data", "information", "results", "findings",
        "paper", "article", "approach", "approaches", "method", "methods",
        "implications", "background", "objective", "aim", "conclusions",
        "sample", "samples", "respondents", "participants", "dataset", "datasets",
    }
    stopwords = list(ENGLISH_STOP_WORDS.union(extra_stopwords))

    vectorizer_model = build_vectorizer(stopwords, n_docs)
    representation_model = MaximalMarginalRelevance(diversity=0.6, top_n_words=5)

    topic_model = BERTopic(
        embedding_model=EMBEDDING_MODEL,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        language="multilingual",
        calculate_probabilities=True,
        verbose=True,
    )

    # Fit + fallback for vectorizer issues
    try:
        topics, probs = topic_model.fit_transform(docs)
    except ValueError as e:
        print(f"WARNING: Vectorizer error ({e}); running fallback.")

        fallback_vectorizer = CountVectorizer(
            stop_words=stopwords,
            ngram_range=(1, 3),
            min_df=1,
            max_df=1.0,
        )

        topic_model = BERTopic(
            embedding_model=EMBEDDING_MODEL,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=fallback_vectorizer,
            representation_model=representation_model,
            language="multilingual",
            calculate_probabilities=True,
            verbose=True,
        )

        topics, probs = topic_model.fit_transform(docs)

    df["topic"] = topics

    # Topic info before auto-labeling
    topic_info_raw = topic_model.get_topic_info()
    topic_info_raw.to_csv(out_dir / "topic_info_raw.csv", index=False)

    # Auto-label topics
    try:
        print("Generating automatic topic labels ...")
        topic_labels = topic_model.generate_topic_labels(
            nr_words=5,
            topic_prefix=False,
            separator=", ",
        )
        topic_model.set_topic_labels(topic_labels)
        print("Auto-labels set (topic_model.custom_labels_).")
    except Exception as e:
        print(f"Notice: could not generate automatic topic labels: {e}")

    # Topic info after auto-labeling
    topic_info = topic_model.get_topic_info().copy()
    if getattr(topic_model, "custom_labels_", None) is not None:
        unique_topics = sorted(set(topic_model.topics_))
        mapping = {t: lbl for t, lbl in zip(unique_topics, topic_model.custom_labels_)}
        topic_info["CustomLabel"] = topic_info["Topic"].map(mapping)
    topic_info_short = topic_info.drop(columns=["Representation", "Representative_Docs"])
    topic_info_short.to_csv(out_dir / "topic_info.csv", index=False)

    print(f"Number of topics (including -1): {len(topic_info)}")

    # Hierarchical topics
    real_topics = topic_info[topic_info["Topic"] != -1]
    n_real_topics = len(real_topics)

    if n_real_topics < 2:
        print(f"Too few meaningful topics ({n_real_topics}) for hierarchy; skipping hierarchical analysis.")
    else:
        print("Computing hierarchical topics ...")
        hierarchical_topics = topic_model.hierarchical_topics(docs)
        hierarchical_topics.to_csv(out_dir / "hierarchical_topics_raw.csv", index=False)

        try:
            fig_hier = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
            fig_hier.write_html(out_dir / "topics_hierarchy.html")
        except Exception as e:
            print("Notice: could not save hierarchy plot:", e)

        try:
            fig_docs = topic_model.visualize_hierarchical_documents(
                docs, hierarchical_topics=hierarchical_topics
            )
            fig_docs.write_html(out_dir / "hierarchical_documents.html")
        except Exception as e:
            print("Notice: could not save document hierarchy plot:", e)

    # Topics over time + trends
    print("Computing topics-over-time ...")
    topics_over_time = topic_model.topics_over_time(
        docs,
        years,
        nr_bins=None
    )
    topics_over_time.to_csv(out_dir / "topics_over_time.csv", index=False)

    tot_non_noise = topics_over_time[topics_over_time["Topic"] != -1].copy()
    if not tot_non_noise.empty:
        pivot = tot_non_noise.pivot_table(
            index="Timestamp",
            columns="Topic",
            values="Frequency",
            fill_value=0,
        ).sort_index()

        top_rising, top_falling = compute_linear_trends(pivot, top_n=5)

        print("\nRising topics (Topic-ID, slope):")
        for t, m in top_rising:
            print(f"{t:>3}  {m: .3f}")

        print("\nDeclining topics (Topic-ID, slope):")
        for t, m in top_falling:
            print(f"{t:>3}  {m: .3f}")

    # Optional: Reduced (macro) topics
    print("Reducing topics (optional macro-themes) ...")
    try:
        n_topics_effective = max(0, len(topic_info) - 1)
        n_macro_topics = min(8, max(3, n_topics_effective)) if n_topics_effective > 0 else 0

        if n_macro_topics >= 3:
            new_topics, new_probs = topic_model.reduce_topics(docs, nr_topics=n_macro_topics)
            df["macro_topic"] = new_topics
            macro_info = topic_model.get_topic_info()
            macro_info.to_csv(out_dir / "macro_topic_info.csv", index=False)
        else:
            print("Too few topics for a meaningful reduction; skipping macro topics.")
    except Exception as e:
        print("Notice: could not reduce topics:", e)

    # Export documents with topics
    df.to_csv(out_dir / "docs_with_topics_and_macro_topics.csv", index=False)
    print(f"Done. Results in {out_dir.resolve()}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    config = DEFAULT_CONFIG.copy()

    if args.config:
        yaml_config = load_yaml_config(args.config)
        config.update(yaml_config)
        print(f"YAML configuration loaded from: {args.config}")

    apply_config_overrides(config)

    print("Active configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Load data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"CSV not found: {DATA_PATH}")

    print("[1/2] Loading abstracts.csv ...")

    df = pd.read_csv(
        DATA_PATH,
        sep=r"\$\$",
        engine="python",
        header=0
    )

    required_cols = {"title", "abstract", "year", "source"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV needs columns: {required_cols}, found: {set(df.columns)}")

    # Run BERTopic once over the full dataset
    print("[2/2] Running BERTopic across all disciplines ...")
    run_bertopic_all(df)


if __name__ == "__main__":
    main()
