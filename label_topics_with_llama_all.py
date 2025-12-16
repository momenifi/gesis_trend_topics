#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
from pathlib import Path

import pandas as pd
from llama_cpp import Llama

# -------------------------------
# CONFIG
# -------------------------------

MODEL_PATH = "test/llama-3.1-8b-instruct-q4_K_M.gguf"
OUTPUT_BASE_DIR = "outputs_by_cluster"

# Welche Cluster sollen gelabelt werden?
CLUSTERS = ["CS", "SS", "BIO", "OTHER"]

# Max topics to label per cluster (None = alle)
MAX_TOPICS = None  # z.B. 20 zum Testen

# Beschreibung pro Cluster für den Prompt
CLUSTER_DOMAIN_DESCRIPTION = {
    "CS": (
        "computer science, machine learning, natural language processing, "
        "information retrieval, data science"
    ),
    "SS": (
        "social sciences, survey research, political science, sociology, psychology, "
        "communication research"
    ),
    "BIO": (
        "biomedical and health research, medicine, public health, neuroscience, "
        "epidemiology"
    ),
    "OTHER": (
        "interdisciplinary and mixed research domains across various fields"
    ),
}


# -------------------------------
# Helper: clean model output
# -------------------------------
def clean_label_text(text: str) -> str:
    """Post-processes the raw LLM output into a clean label string."""
    if not isinstance(text, str):
        return ""
    # nur erste Zeile
    text = text.strip().split("\n")[0]
    # führende Bulletpoints / Nummern entfernen
    text = re.sub(r"^[\-\*\d\.\)\s]+", "", text)
    # Anführungszeichen entfernen
    text = text.strip(" '\"")
    return text


# -------------------------------
# Build prompt for one topic
# -------------------------------
def build_prompt_for_topic(topic_id, custom_name, name, count, domain_description: str) -> str:
    """Baue den User-Prompt für ein einzelnes Topic."""
    # Fallback: sinnvollen Descriptor wählen
    descriptor = custom_name if isinstance(custom_name, str) and custom_name.strip() else name
    descriptor = descriptor if isinstance(descriptor, str) else ""

    return f"""
You are an expert in {domain_description} and scientometrics.

You see a topic from a corpus of research abstracts.
The topic is described by the following key terms:
- {descriptor}

This topic has {count} documents.

TASK:
1. Give ONE short, human-readable label for this topic.
2. The label MUST be:
   - a noun phrase (no full sentence, no verb like "modeling", "assessing", "using")
   - 3 to 6 words long
   - informative and specific enough for a scientific figure caption
   - without quotes, numbering, or explanations.

Output ONLY the label.
""".strip()


# -------------------------------
# Label all topics for a cluster
# -------------------------------
def label_topics_for_cluster(llm: Llama, cluster: str):
    cluster_dir = Path(OUTPUT_BASE_DIR) / f"cluster_{cluster}"
    input_csv = cluster_dir / "topic_info.csv"
    output_csv = cluster_dir / "topic_info_labeled.csv"

    if not input_csv.exists():
        print(f"[{cluster}] topic_info.csv not found at {input_csv}, skipping.")
        return

    print(f"\n=== Cluster {cluster}: labeling topics ===")
    print(f"Loading topics from: {input_csv}")

    df = pd.read_csv(input_csv)

    if "Topic" not in df.columns:
        raise ValueError(f"[{cluster}] Column 'Topic' is missing from topic_info.csv")

    # LLM_Label-Spalte hinzufügen, falls nicht vorhanden
    if "LLM_Label" not in df.columns:
        df["LLM_Label"] = ""

    # Nur echte Topics (ohne -1)
    topic_rows = df[df["Topic"] != -1].copy()

    if MAX_TOPICS is not None:
        topic_rows = topic_rows.head(MAX_TOPICS)

    domain_description = CLUSTER_DOMAIN_DESCRIPTION.get(
        cluster,
        CLUSTER_DOMAIN_DESCRIPTION["OTHER"],
    )

    print(f"[{cluster}] Labeling {len(topic_rows)} topics with Llama 3.1...")

    for idx, row in topic_rows.iterrows():
        topic_id = row["Topic"]

        # Überspringen, wenn schon Label vorhanden
        if isinstance(df.loc[idx, "LLM_Label"], str) and df.loc[idx, "LLM_Label"].strip():
            continue

        custom_name = row.get("CustomName", "")
        name = row.get("Name", "")
        count = row.get("Count", "")

        descriptor = custom_name if isinstance(custom_name, str) and custom_name.strip() else name
        print(f"\n[{cluster}] === Topic {topic_id} ===")
        print(f"Descriptor: {descriptor}")

        prompt = build_prompt_for_topic(
            topic_id=topic_id,
            custom_name=custom_name,
            name=name,
            count=count,
            domain_description=domain_description,
        )

        # Chat-Aufruf
        resp = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You label research topics for scientific figures. "
                        "Always output ONE short noun phrase (3–6 words)."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=32,
            temperature=0.1,
        )

        raw_text = resp["choices"][0]["message"]["content"]
        label = clean_label_text(raw_text)

        print(f"[{cluster}] LLM label: {label}")
        df.loc[idx, "LLM_Label"] = label

        # Zwischenspeichern
        df.to_csv(output_csv, index=False)

    print(f"[{cluster}] Done. Saved labeled topics to: {output_csv}")


# -------------------------------
# MAIN
# -------------------------------
def main():
    # Llama-Modell einmal laden und für alle Cluster wiederverwenden
    print(f"Loading Llama model from: {MODEL_PATH}")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_threads=8,      # ggf. an deine CPU anpassen
        n_gpu_layers=-1,  # -1 = volle GPU-Offload (wenn verfügbar), 0 = nur CPU
    )

    for cluster in CLUSTERS:
        label_topics_for_cluster(llm, cluster)

    print("\nAll done.")


if __name__ == "__main__":
    main()
