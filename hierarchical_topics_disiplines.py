#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BERTopic-Analyse getrennt nach Disziplin-Clustern (SS, CS, BIO, OTHER).

CSV-Format (mit $$ als Separator):
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
from bertopic.representation import KeyBERTInspired
from bertopic.representation import MaximalMarginalRelevance
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
import umap
import hdbscan


# ---------------------------------------------------------------------
# KONFIGURATION
# ---------------------------------------------------------------------
DATA_PATH = "abstracts.csv"              # deine Datei mit $$-Separator
OUTPUT_BASE_DIR = "outputs_by_cluster"   # Basisordner für alle Ergebnisordner

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
TITLE_COLUMN = "title"
KEYWORD_COLUMN = "keyword"
KEYWORD_WEIGHT = 2

MIN_CLUSTER_SIZE = 10
N_COMPONENTS = 5
N_NEIGHBORS = 15
RANDOM_STATE = 42

# Untergrenze, ab wann sich Topic Modeling für ein Cluster überhaupt lohnt
MIN_DOCS_FOR_CLUSTER = 10

DEFAULT_CONFIG = {
    "data_path": DATA_PATH,
    "output_base_dir": OUTPUT_BASE_DIR,
    "embedding_model": EMBEDDING_MODEL,
    "title_column": TITLE_COLUMN,
    "keyword_column": KEYWORD_COLUMN,
    "keyword_weight": KEYWORD_WEIGHT,
    "min_cluster_size": MIN_CLUSTER_SIZE,
    "n_components": N_COMPONENTS,
    "n_neighbors": N_NEIGHBORS,
    "random_state": RANDOM_STATE,
    "min_docs_for_cluster": MIN_DOCS_FOR_CLUSTER,
}


def load_yaml_config(config_path: str) -> dict:
    """Lädt YAML-Konfiguration und stellt sicher, dass ein Mapping vorliegt."""
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("Konfigurationsdatei muss ein YAML-Mapping enthalten.")

    return data


def apply_config_overrides(config: dict):
    """Wendet Konfigurationswerte auf die Modul-Globals an."""
    global DATA_PATH, OUTPUT_BASE_DIR, EMBEDDING_MODEL
    global TITLE_COLUMN, KEYWORD_COLUMN, KEYWORD_WEIGHT
    global MIN_CLUSTER_SIZE, N_COMPONENTS, N_NEIGHBORS, RANDOM_STATE
    global MIN_DOCS_FOR_CLUSTER

    DATA_PATH = config.get("data_path", DATA_PATH)
    OUTPUT_BASE_DIR = config.get("output_base_dir", OUTPUT_BASE_DIR)
    EMBEDDING_MODEL = config.get("embedding_model", EMBEDDING_MODEL)
    TITLE_COLUMN = config.get("title_column", TITLE_COLUMN)
    KEYWORD_COLUMN = config.get("keyword_column", KEYWORD_COLUMN)
    KEYWORD_WEIGHT = int(config.get("keyword_weight", KEYWORD_WEIGHT))

    MIN_CLUSTER_SIZE = int(config.get("min_cluster_size", MIN_CLUSTER_SIZE))
    N_COMPONENTS = int(config.get("n_components", N_COMPONENTS))
    N_NEIGHBORS = int(config.get("n_neighbors", N_NEIGHBORS))
    RANDOM_STATE = int(config.get("random_state", RANDOM_STATE))
    MIN_DOCS_FOR_CLUSTER = int(config.get("min_docs_for_cluster", MIN_DOCS_FOR_CLUSTER))


# ---------------------------------------------------------------------
# HILFSFUNKTIONEN
# ---------------------------------------------------------------------
def simple_clean(text: str) -> str:
    """Einfache Textreinigung für Abstracts."""
    if not isinstance(text, str):
        return ""
    text = text.strip()

    # Falls das Abstract in {" ... "} eingepackt ist
    if text.startswith('{"') and text.endswith('"}'):
        text = text[2:-2]

    text = text.replace("\\n", " ")
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_weighted_text(title: str, abstract: str, keywords: str, keyword_weight: int) -> str:
    """Kombiniert Title+Abstract+Keywords, gewichtet Keywords durch Wiederholung."""
    parts = []
    if isinstance(title, str) and title.strip():
        parts.append(title.strip())
    if isinstance(abstract, str) and abstract.strip():
        parts.append(abstract.strip())
    if isinstance(keywords, str) and keywords.strip():
        parts.append(keywords.strip())
        if keyword_weight > 1:
            parts.extend([keywords.strip()] * (keyword_weight - 1))
    combined = " ".join(parts)
    return simple_clean(combined)


def compute_linear_trends(count_table: pd.DataFrame, top_n: int = 10):
    """Berechnet lineare Trends je Topic aus year x topic-Tabelle."""
    slopes = []
    x = np.arange(len(count_table.index), dtype=float)

    for topic in count_table.columns:
        y = count_table[topic].values.astype(float)
        if len(y) < 2 or np.all(y == y[0]):
            m = 0.0
        else:
            m, _ = np.polyfit(x, y, 1)
        slopes.append((topic, m))

    slopes_sorted = sorted(slopes, key=lambda x: x[1], reverse=True)
    top_rising = slopes_sorted[:top_n]
    top_falling = slopes_sorted[-top_n:]
    return top_rising, top_falling


def map_middle_group_to_cluster(mg: str) -> str:
    """Basismapping: Scopus middle_group -> Cluster (SS, CS, BIO, OTHER)"""
    if not isinstance(mg, str):
        return "OTHER"

    mg = mg.strip()

    # Social Sciences Cluster
    SS_set = {
        "Social Sciences",
        "Psychology",
        "Economics, Econometrics and Finance",
        "Business, Management and Accounting",
        "Arts and Humanities",
        "Education",
    }

    # Computer / Data Science Cluster
    CS_set = {
        "Computer Science",
        "Mathematics",
        "Decision Sciences",
        "Engineering",
    }

    # Biomedical / Health Cluster
    BIO_set = {
        "Medicine",
        "Neuroscience",
        "Biochemistry, Genetics and Molecular Biology",
        "Health Professions",
        "Immunology and Microbiology",
    }

    if mg in SS_set:
        return "SS"
    if mg in CS_set:
        return "CS"
    if mg in BIO_set:
        return "BIO"
    return "OTHER"


def classify_document(middle_group_value: str, abstract: str) -> str:
    """
    Ordnet einen Artikel einem der vier Cluster zu: 'SS', 'CS', 'BIO', 'OTHER'.
    """
    if not isinstance(middle_group_value, str):
        middle_group_value = ""
    if not isinstance(abstract, str):
        abstract = ""

    groups = [g.strip() for g in re.split(r"[;|,]", middle_group_value) if g.strip()]
    abs_low = abstract.lower()

    # Keywords für Feintuning
    ss_keywords = [
        "attitudes", "perceptions", "media", "voters", "election", "democracy",
        "public opinion", "cohort", "inequality", "trust", "respondents",
        "survey", "panel", "questionnaire", "social sciences", "political"
    ]
    cs_keywords = [
        "language model", "large language model", "llm", "bert",
        "embedding", "neural", "deep learning", "machine learning",
        "transformer", "algorithm", "classification", "retrieval",
        "entity extraction", "information retrieval", "nlp", "corpus"
    ]

    mapped = [map_middle_group_to_cluster(g) for g in groups]

    # Explizite Zuweisung
    if "SS" in mapped:
        return "SS"
    if "CS" in mapped:
        # Engineering als CS nur, wenn Text nicht klar SS-artig ist
        if any(k in abs_low for k in ss_keywords):
            return "SS"
        return "CS"
    if "BIO" in mapped:
        return "BIO"

    # Kein klares Mapping → über Keywords
    if any(k in abs_low for k in cs_keywords):
        return "CS"
    if any(k in abs_low for k in ss_keywords):
        return "SS"

    return "OTHER"


def build_vectorizer(stopwords, n_docs: int) -> CountVectorizer:
    """
    Erzeugt einen CountVectorizer mit dynamischem min_df und robusten Einstellungen.
    """
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
        description="Führt die BERTopic-Pipeline nach Disziplin-Clustern aus."
    )
    parser.add_argument(
        "-c",
        "--config",
        help=(
            "Pfad zu einer YAML-Konfigurationsdatei, die Standardwerte wie "
            "data_path oder embedding_model überschreibt. Beispiel: "
            "config.example.yaml"
        ),
    )
    return parser.parse_args()


def run_bertopic_for_cluster(df_cluster: pd.DataFrame, cluster_label: str):
    """
    Führt die komplette BERTopic-Pipeline für einen Disziplin-Cluster aus.
    Ergebnisse werden in OUTPUT_BASE_DIR/cluster_<label>/ geschrieben.
    """
    if df_cluster.empty:
        print(f"\n==== Cluster {cluster_label}: keine Dokumente, überspringe. ====")
        return

    print(f"\n==== Cluster {cluster_label}: n={len(df_cluster)} Dokumente ====")

    if len(df_cluster) < MIN_DOCS_FOR_CLUSTER:
        print(f"[{cluster_label}] Weniger als {MIN_DOCS_FOR_CLUSTER} Dokumente – Topic Modeling wird übersprungen.")
        return

    out_dir = Path(OUTPUT_BASE_DIR) / f"cluster_{cluster_label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Daten vorbereiten
    df = df_cluster.copy()
    df = df.dropna(subset=["abstract", "year"]).copy()
    df["year_raw"] = df["year"]
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    df = df.sort_values("year").reset_index(drop=True)

    df["clean_text"] = df.apply(
        lambda row: build_weighted_text(
            row.get(TITLE_COLUMN, ""),
            row.get("abstract", ""),
            row.get(KEYWORD_COLUMN, ""),
            KEYWORD_WEIGHT,
        ),
        axis=1,
    )
    docs = df["clean_text"].tolist()
    years = df["year"].tolist()
    n_docs = len(df)

    print(f"[{cluster_label}] Jahre im Datensatz: {df['year'].min()} – {df['year'].max()} (n={n_docs})")
    print(f"[{cluster_label}] Nutze EMBEDDING_MODEL in BERTopic: {EMBEDDING_MODEL}")

    # 2) BERTopic-Konfiguration
    print(f"[{cluster_label}] Fitte BERTopic-Modell ...")

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
    
    # ML-/NLP-spezifische Stopwords nur für CS
    ml_stopwords = set()
    if cluster_label == "CS":
        ml_stopwords = {
            "model", "models", "learning", "machine", "neural",
            "embedding", "embeddings",
            "classification", "classifications", "classifier", "classifiers",
            "token", "tokens",
            "nlp", "natural", "processing", "transformer", "transformers",
            "bert", "gpt"
        }
    stopwords = list(ENGLISH_STOP_WORDS.union(stopwords).union(ml_stopwords))
    
    vectorizer_model = build_vectorizer(stopwords, n_docs)
    #representation_model = KeyBERTInspired()
    representation_model = MaximalMarginalRelevance(diversity=0.6, top_n_words=5)
    #representation_model = [
    #KeyBERTInspired(),
    #MaximalMarginalRelevance(diversity=0.4, top_n_words=5)
    #]
    

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

    # 3) Fit + Fallback für Vectorizer-Probleme
    try:
        topics, probs = topic_model.fit_transform(docs)
    except ValueError as e:
        print(f"[{cluster_label}] WARNUNG: Fehler im Vectorizer ({e}), starte vollständigen Fallback ...")

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
    if probs is None:
        df["topic_prob"] = np.nan
    else:
        probs_arr = np.asarray(probs)
        if probs_arr.ndim == 1:
            df["topic_prob"] = probs_arr
        else:
            df["topic_prob"] = np.max(probs_arr, axis=1)

    # 3b) Topic-Info VOR Auto-Labeling
    topic_info_raw = topic_model.get_topic_info()
    topic_info_raw.to_csv(out_dir / "topic_info_raw.csv", index=False)

    # 3c) Auto-Labels generieren und setzen
    try:
        print(f"[{cluster_label}] Erzeuge automatische Topic-Labels ...")
        topic_labels = topic_model.generate_topic_labels(
            nr_words=5,
            topic_prefix=False,      # z.B. nur Wörter, ohne "0_"
            separator=", ",
        )
        topic_model.set_topic_labels(topic_labels)
        print(f"[{cluster_label}] Auto-Labels gesetzt (topic_model.custom_labels_).")
    except Exception as e:
        print(f"[{cluster_label}] Hinweis: automatische Topic-Labels konnten nicht erzeugt werden: {e}")

    # 3d) Topic-Info NACH Auto-Labeling
    topic_info = topic_model.get_topic_info().copy()
    # optional: CustomLabel-Spalte explizit aus custom_labels_ schreiben
    if getattr(topic_model, "custom_labels_", None) is not None:
        unique_topics = sorted(set(topic_model.topics_))
        mapping = {t: lbl for t, lbl in zip(unique_topics, topic_model.custom_labels_)}
        topic_info["CustomLabel"] = topic_info["Topic"].map(mapping)
    topic_info_short = topic_info.drop(columns=["Representation", "Representative_Docs"])
    topic_info_short.to_csv(out_dir / "topic_info.csv", index=False)

    print(f"[{cluster_label}] Anzahl Topics (inkl. -1): {len(topic_info)}")

    # Topic-Label pro Dokument zuordnen (falls vorhanden)
    label_col = None
    for col in ("CustomLabel", "Name"):
        if col in topic_info.columns:
            label_col = col
            break
    if label_col:
        topic_label_map = dict(zip(topic_info["Topic"], topic_info[label_col]))
        df["topic_label"] = df["topic"].map(topic_label_map)

    # 4) Hierarchische Topics
    real_topics = topic_info[topic_info["Topic"] != -1]
    n_real_topics = len(real_topics)

    if n_real_topics < 2:
        print(f"[{cluster_label}] Zu wenige sinnvolle Topics ({n_real_topics}) "
              f"für Hierarchie – überspringe hierarchische Analyse.")
    else:
        print(f"[{cluster_label}] Berechne hierarchische Topics ...")
        hierarchical_topics = topic_model.hierarchical_topics(docs)
        hierarchical_topics.to_csv(out_dir / "hierarchical_topics_raw.csv", index=False)

        try:
            fig_hier = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
            fig_hier.write_html(out_dir / "topics_hierarchy.html")
        except Exception as e:
            print(f"[{cluster_label}] Hinweis: Konnte Hierarchie-Plot nicht speichern:", e)

        try:
            fig_docs = topic_model.visualize_hierarchical_documents(
                docs, hierarchical_topics=hierarchical_topics
            )
            fig_docs.write_html(out_dir / "hierarchical_documents.html")
        except Exception as e:
            print(f"[{cluster_label}] Hinweis: Konnte Dokumenten-Hierarchie nicht speichern:", e)

    # 5) Topics over time + Trends
    print(f"[{cluster_label}] Berechne Topics-over-time ...")
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

        print(f"\n[{cluster_label}] ▲ Aufsteigende Topics (Topic-ID, Steigung):")
        for t, m in top_rising:
            print(f"{t:>3}  {m: .3f}")

        print(f"\n[{cluster_label}] ▼ Abnehmende Topics (Topic-ID, Steigung):")
        for t, m in top_falling:
            print(f"{t:>3}  {m: .3f}")

    # 6) Optional: Reduced (Makro-)Topics
    print(f"[{cluster_label}] Reduziere Topics (optional Makro-Themen) ...")
    try:
        n_topics_effective = max(0, len(topic_info) - 1)  # -1 für Topic -1
        N_MACRO_TOPICS = min(8, max(3, n_topics_effective)) if n_topics_effective > 0 else 0

        if N_MACRO_TOPICS >= 3:
            new_topics, new_probs = topic_model.reduce_topics(docs, nr_topics=N_MACRO_TOPICS)
            df["macro_topic"] = new_topics
            macro_info = topic_model.get_topic_info()
            macro_info.to_csv(out_dir / "macro_topic_info.csv", index=False)
        else:
            print(f"[{cluster_label}] Zu wenige Topics für eine sinnvolle Reduktion – Makro-Themen werden übersprungen.")
    except Exception as e:
        print(f"[{cluster_label}] Hinweis: Konnte Topics nicht reduzieren:", e)

    # 7) Export Dokumente
    df.to_csv(out_dir / "docs_with_topics_and_macro_topics.csv", index=False)
    # Publikationsliste mit Topic-Zuweisung (pro Cluster)
    cols = [
        TITLE_COLUMN, "abstract", "year", "source", KEYWORD_COLUMN,
        "middle_group", "cluster", "topic", "topic_label", "topic_prob",
    ]
    export_cols = [c for c in cols if c in df.columns]
    df[export_cols].to_csv(out_dir / "publications_with_topics.csv", index=False)
    df[export_cols].to_excel(out_dir / "publications_with_topics.xlsx", index=False)
    print(f"[{cluster_label}] Fertig. Ergebnisse in {out_dir.resolve()}")


# ---------------------------------------------------------------------
# HAUPTFUNKTION
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    config = DEFAULT_CONFIG.copy()

    if args.config:
        yaml_config = load_yaml_config(args.config)
        config.update(yaml_config)
        print(f"YAML-Konfiguration geladen aus: {args.config}")

    apply_config_overrides(config)

    print("Aktive Konfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 1) Daten laden
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"CSV nicht gefunden: {DATA_PATH}")

    print("[1/3] Lade abstracts.csv ...")

    df = pd.read_csv(
        DATA_PATH,
        sep=r"\$\$",
        engine="python",
        header=0
    )

    required_cols = {TITLE_COLUMN, "abstract", "year", "source", "middle_group"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV braucht Spalten: {required_cols}, gefunden: {set(df.columns)}")
    if KEYWORD_COLUMN not in df.columns:
        print(f"Warnung: Spalte '{KEYWORD_COLUMN}' fehlt, verwende leere Keywords.")
        df[KEYWORD_COLUMN] = ""

    # 2) Disziplin-Cluster zuordnen
    print("[2/3] Ordne Disziplin-Cluster zu ...")
    df["cluster"] = df.apply(
        lambda row: classify_document(
            row.get("middle_group", ""),
            row.get("abstract", "")
        ),
        axis=1
    )

    print("Verteilung cluster:")
    print(df["cluster"].value_counts())

    # 3) Pro Cluster BERTopic laufen lassen
    print("\n[3/3] Starte BERTopic pro Cluster ...")
    for cluster_label in ["SS", "CS", "BIO", "OTHER"]:
        df_cluster = df[df["cluster"] == cluster_label]
        run_bertopic_for_cluster(df_cluster, cluster_label)

    # 4) Zusätzlich Topic Modeling über alle Disziplinen gemeinsam
    print("\n[4/3] Starte BERTopic über alle Disziplinen ...")
    run_bertopic_for_cluster(df, "ALL")


if __name__ == "__main__":
    main()
