#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import re
from pathlib import Path

import pandas as pd
from llama_cpp import Llama

# -------------------------------
# CONFIG
# -------------------------------

MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "llama-3.1-8b-instruct-q4_K_M.gguf")
OUTPUT_BASE_DIR = "outputs_by_cluster"

# Welche Cluster sollen gelabelt werden?
CLUSTERS = ["CS", "SS", "BIO", "OTHER", "ALL"]

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


def parse_multi_labels(text: str, max_labels: int = 3) -> list[str]:
    """Parse up to max_labels from LLM output (bullets/lines/semicolon separated)."""
    if not isinstance(text, str):
        return []
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return []
    if len(lines) == 1 and ";" in lines[0]:
        parts = [p.strip() for p in lines[0].split(";")]
    else:
        parts = lines
    labels = []
    for part in parts:
        cleaned = clean_label_text(part)
        if cleaned:
            labels.append(cleaned)
        if len(labels) >= max_labels:
            break
    return labels


def pick_descriptor(row: pd.Series) -> str:
    """Pick the best descriptor for a topic (prefer custom labels)."""
    for col in ("CustomLabel", "CustomName", "Name"):
        value = row.get(col, "")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


# -------------------------------
# Build prompt for one topic
# -------------------------------
def build_prompt_for_topic(topic_id, descriptor, name, count, domain_description: str) -> str:
    """Baue den User-Prompt für ein einzelnes Topic."""
    # Fallback: sinnvollen Descriptor wählen
    descriptor = descriptor if isinstance(descriptor, str) else ""
    name = name if isinstance(name, str) else ""

    return f"""
You are an expert in {domain_description} and scientometrics.

You see a topic from a corpus of research abstracts.
The topic is described by the following key terms:
- {descriptor}
{f"- Alternative terms: {name}" if name and name != descriptor else ""}

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


def estimate_tokens(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return max(1, len(text) // 4)


def format_doc_snippet(title: str, abstract: str, keywords: str, max_chars: int = 600) -> str:
    title = title if isinstance(title, str) else ""
    abstract = abstract if isinstance(abstract, str) else ""
    keywords = keywords if isinstance(keywords, str) else ""
    snippet = f"Title: {title}\nKeywords: {keywords}\nAbstract: {abstract}"
    if len(snippet) > max_chars:
        snippet = snippet[: max_chars - 3].rstrip() + "..."
    return snippet


def build_prompt_for_topic_with_docs(
    topic_id,
    descriptor,
    name,
    count,
    domain_description: str,
    doc_snippets: list[str],
    max_labels: int,
) -> str:
    """Build prompt using representative documents from the topic."""
    descriptor = descriptor if isinstance(descriptor, str) else ""
    name = name if isinstance(name, str) else ""
    docs_block = "\n\n".join(f"Doc {i + 1}:\n{doc}" for i, doc in enumerate(doc_snippets))
    return f"""
You are an expert in {domain_description} and scientometrics.

You see a topic from a corpus of research abstracts.
The topic is described by the following key terms:
- {descriptor}
{f"- Alternative terms: {name}" if name and name != descriptor else ""}

This topic has {count} documents. Below are representative documents from this topic:
{docs_block}

TASK:
1. Propose up to {max_labels} short, human-readable labels for this topic.
2. Each label MUST be:
   - a noun phrase (no full sentence)
   - 3 to 6 words long
   - grounded ONLY in the documents and key terms shown above
3. Output labels as a list, one per line. No explanations.
""".strip()


# -------------------------------
# Label all topics for a cluster
# -------------------------------
def label_topics_for_cluster(
    llm: Llama | None,
    cluster: str,
    max_topics: int | None = None,
    temperature: float = 0.1,
    only_topics_over_time: bool = False,
    model_path: str | None = None,
    n_ctx: int = 4096,
    n_threads: int = 8,
    n_gpu_layers: int = -1,
    doc_sample_fraction: float = 1.0,
    max_docs_per_topic: int = 50,
    max_labels: int = 3,
    prompt_token_budget: int | None = None,
):
    cluster_dir = Path(OUTPUT_BASE_DIR) / f"cluster_{cluster}"
    input_csv = cluster_dir / "topic_info.csv"
    output_csv = cluster_dir / "topic_info_labeled.csv"
    topics_over_time = cluster_dir / "topics_over_time.csv"
    labeled_over_time = cluster_dir / "topics_over_time_labeled.csv"

    if only_topics_over_time:
        if not output_csv.exists():
            print(f"[{cluster}] topic_info_labeled.csv not found at {output_csv}, skipping.")
            return
        if not topics_over_time.exists():
            print(f"[{cluster}] topics_over_time.csv not found at {topics_over_time}, skipping.")
            return
        df = pd.read_csv(output_csv)
        if "Topic" not in df.columns:
            print(f"[{cluster}] Missing Topic column in {output_csv}, skipping.")
            return
        if "LLM_Label" not in df.columns:
            df["LLM_Label"] = ""
        if "LLM_Labels" not in df.columns:
            df["LLM_Labels"] = ""

        tot = pd.read_csv(topics_over_time)
        if "Topic" not in tot.columns:
            print(f"[{cluster}] Missing Topic column in {topics_over_time}, skipping.")
            return

        missing_topics = set(tot["Topic"].unique()) - set(df["Topic"].unique())
        missing_topics.discard(-1)
        if missing_topics:
            topic_info_csv = cluster_dir / "topic_info.csv"
            if topic_info_csv.exists():
                df_info = pd.read_csv(topic_info_csv)
                if "Topic" in df_info.columns:
                    df = pd.concat([df, df_info[df_info["Topic"].isin(missing_topics)]])
                    df = df.drop_duplicates(subset=["Topic"], keep="first")

        missing_labels = df[
            df["Topic"].isin(tot["Topic"].unique())
            & (df["Topic"] != -1)
            & (~df["LLM_Label"].astype(str).str.strip().astype(bool))
        ]
        if not missing_labels.empty:
            if llm is None:
                if not model_path:
                    print(f"[{cluster}] Missing labels but no model path provided, skipping LLM.")
                else:
                    print(f"[{cluster}] Loading Llama model from: {model_path}")
                    llm = Llama(
                        model_path=model_path,
                        n_ctx=n_ctx,
                        n_threads=n_threads,
                        n_gpu_layers=n_gpu_layers,
                    )
            if llm is not None:
                domain_description = CLUSTER_DOMAIN_DESCRIPTION.get(
                    cluster,
                    CLUSTER_DOMAIN_DESCRIPTION["OTHER"],
                )
                for idx, row in missing_labels.iterrows():
                    topic_id = row.get("Topic")
                    descriptor = pick_descriptor(row)
                    name = row.get("Name", "")
                    count = row.get("Count", "")
                    prompt = build_prompt_for_topic(
                        topic_id=topic_id,
                        descriptor=descriptor,
                        name=name,
                        count=count,
                        domain_description=domain_description,
                    )
                    resp = llm.create_chat_completion(
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You label research topics for scientific figures. "
                                    "Always output ONE short noun phrase (3-6 words)."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=32,
                        temperature=temperature,
                    )
                    raw_text = resp["choices"][0]["message"]["content"]
                    labels = parse_multi_labels(raw_text, max_labels=max_labels)
                    if not labels:
                        labels = [descriptor or name or f"Topic {topic_id}"]
                    df.loc[df["Topic"] == topic_id, "LLM_Label"] = labels[0]
                    df.loc[df["Topic"] == topic_id, "LLM_Labels"] = "; ".join(labels)

                df.to_csv(output_csv, index=False)

        label_map = dict(zip(df["Topic"], df["LLM_Label"]))
        tot["LLM_Label"] = tot["Topic"].map(label_map)
        tot.to_csv(labeled_over_time, index=False)
        print(f"[{cluster}] Saved labeled topics_over_time to: {labeled_over_time}")
        return

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
    if "LLM_Labels" not in df.columns:
        df["LLM_Labels"] = ""

    # Nur echte Topics (ohne -1)
    topic_rows = df[df["Topic"] != -1].copy()

    if max_topics is not None:
        topic_rows = topic_rows.head(max_topics)

    domain_description = CLUSTER_DOMAIN_DESCRIPTION.get(
        cluster,
        CLUSTER_DOMAIN_DESCRIPTION["OTHER"],
    )

    print(f"[{cluster}] Labeling {len(topic_rows)} topics with Llama 3.1...")

    docs_csv = cluster_dir / "publications_with_topics.csv"
    if not docs_csv.exists():
        docs_csv = cluster_dir / "docs_with_topics_and_macro_topics.csv"
    docs_df = pd.read_csv(docs_csv) if docs_csv.exists() else None

    for idx, row in topic_rows.iterrows():
        topic_id = row["Topic"]

        # Überspringen, wenn schon Label vorhanden
        if isinstance(df.loc[idx, "LLM_Label"], str) and df.loc[idx, "LLM_Label"].strip():
            continue

        descriptor = pick_descriptor(row)
        name = row.get("Name", "")
        count = row.get("Count", "")
        print(f"\n[{cluster}] === Topic {topic_id} ===")
        print(f"Descriptor: {descriptor}")

        if docs_df is not None and "topic" in docs_df.columns:
            docs_topic = docs_df[docs_df["topic"] == topic_id].copy()
            n_total = len(docs_topic)
            if n_total > 0:
                n_sample = max(1, int(round(n_total * doc_sample_fraction)))
                if "topic_prob" in docs_topic.columns:
                    docs_topic = docs_topic.sort_values(
                        "topic_prob", ascending=False
                    ).head(min(n_sample, n_total))
                else:
                    docs_topic = docs_topic.sample(n=min(n_sample, n_total), random_state=42)
                if max_docs_per_topic and len(docs_topic) > max_docs_per_topic:
                    docs_topic = docs_topic.head(max_docs_per_topic)
                budget = prompt_token_budget
                if budget is None:
                    budget = max(512, n_ctx - 512)
                else:
                    budget = min(budget, max(512, n_ctx - 512))
                doc_snippets = []
                used_tokens = 0
                for _, row_doc in docs_topic.iterrows():
                    snippet = format_doc_snippet(
                        row_doc.get("title", ""),
                        row_doc.get("abstract", ""),
                        row_doc.get("keyword", ""),
                    )
                    snippet_tokens = estimate_tokens(snippet)
                    if used_tokens + snippet_tokens > budget:
                        break
                    doc_snippets.append(snippet)
                    used_tokens += snippet_tokens
                prompt = build_prompt_for_topic_with_docs(
                    topic_id=topic_id,
                    descriptor=descriptor,
                    name=name,
                    count=count,
                    domain_description=domain_description,
                    doc_snippets=doc_snippets,
                    max_labels=max_labels,
                )
            else:
                prompt = build_prompt_for_topic(
                    topic_id=topic_id,
                    descriptor=descriptor,
                    name=name,
                    count=count,
                    domain_description=domain_description,
                )
        else:
            prompt = build_prompt_for_topic(
                topic_id=topic_id,
                descriptor=descriptor,
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
                        "Always output ONE short noun phrase (3-6 words)."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=32,
            temperature=temperature,
        )

        raw_text = resp["choices"][0]["message"]["content"]
        labels = parse_multi_labels(raw_text, max_labels=max_labels)
        if not labels:
            labels = [descriptor or name or f"Topic {topic_id}"]

        print(f"[{cluster}] LLM labels: {labels}")
        df.loc[idx, "LLM_Label"] = labels[0]
        df.loc[idx, "LLM_Labels"] = "; ".join(labels)

        # Zwischenspeichern
        df.to_csv(output_csv, index=False)

    print(f"[{cluster}] Done. Saved labeled topics to: {output_csv}")

    if topics_over_time.exists():
        tot = pd.read_csv(topics_over_time)
        if "Topic" in tot.columns:
            label_map = dict(zip(df["Topic"], df["LLM_Label"]))
            tot["LLM_Label"] = tot["Topic"].map(label_map)
            tot.to_csv(labeled_over_time, index=False)
            print(f"[{cluster}] Saved labeled topics_over_time to: {labeled_over_time}")


# -------------------------------
# MAIN
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Label BERTopic topics using Llama.")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--output-base-dir", default=OUTPUT_BASE_DIR)
    parser.add_argument("--clusters", default=",".join(CLUSTERS))
    parser.add_argument("--max-topics", type=int, default=MAX_TOPICS)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--doc-sample-fraction", type=float, default=1.0)
    parser.add_argument("--max-docs-per-topic", type=int, default=50)
    parser.add_argument("--max-labels", type=int, default=3)
    parser.add_argument("--prompt-token-budget", type=int, default=None)
    parser.add_argument(
        "--only-topics-over-time",
        action="store_true",
        help="Only create topics_over_time_labeled.csv from existing topic_info_labeled.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    global OUTPUT_BASE_DIR
    OUTPUT_BASE_DIR = args.output_base_dir
    clusters = [c.strip() for c in args.clusters.split(",") if c.strip()]

    llm = None
    if not args.only_topics_over_time:
        # Llama-Modell einmal laden und für alle Cluster wiederverwenden
        print(f"Loading Llama model from: {args.model_path}")
        llm = Llama(
            model_path=args.model_path,
            n_ctx=args.n_ctx,
            n_threads=args.n_threads,      # ggf. an deine CPU anpassen
            n_gpu_layers=args.n_gpu_layers,  # -1 = volle GPU-Offload (wenn verfügbar), 0 = nur CPU
        )

    for cluster in clusters:
        label_topics_for_cluster(
            llm,
            cluster,
            max_topics=args.max_topics,
            temperature=args.temperature,
            only_topics_over_time=args.only_topics_over_time,
            model_path=args.model_path,
            n_ctx=args.n_ctx,
            n_threads=args.n_threads,
            n_gpu_layers=args.n_gpu_layers,
            doc_sample_fraction=args.doc_sample_fraction,
            max_docs_per_topic=args.max_docs_per_topic,
            max_labels=args.max_labels,
            prompt_token_budget=args.prompt_token_budget,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
