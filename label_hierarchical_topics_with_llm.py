#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import ast
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

_LLAMA_CLS = None


def get_llama_cls():
    global _LLAMA_CLS
    if _LLAMA_CLS is None:
        from llama_cpp import Llama as _Llama
        _LLAMA_CLS = _Llama
    return _LLAMA_CLS


MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "llama-3.1-8b-instruct-q4_K_M.gguf")
OUTPUT_BASE_DIR = "outputs_by_cluster"

OPENWEBUI_BASE_URL = os.getenv("OPENWEBUI_BASE_URL", "https://ai-openwebui.gesis.org")
OPENWEBUI_API_KEY = os.getenv("OPENWEBUI_API_KEY")
OPENWEBUI_MODEL = os.getenv("OPENWEBUI_MODEL")

CLUSTERS = ["CS", "SS", "BIO", "OTHER", "ALL"]

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


def clean_label_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip().split("\n")[0]
    text = re.sub(r"^[\-\*\d\.\)\s]+", "", text)
    text = text.strip(" '\"")
    return text


def parse_multi_labels(text: str, max_labels: int = 3) -> list[str]:
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


def list_openwebui_models(base_url: str, api_key: str, timeout: int = 30) -> list[dict]:
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(f"{base_url}/api/models", headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", []) if isinstance(data, dict) else []


def _score_model_id(model_id: str) -> int:
    s = model_id.lower()
    score = 0
    if "o1" in s or "gpt-4o" in s:
        score += 10000
    elif "gpt-4" in s:
        score += 9000
    elif "claude-3.5" in s:
        score += 8000
    elif "claude-3" in s:
        score += 7000
    elif "gemini-1.5-pro" in s:
        score += 6000
    elif "llama-3.1" in s:
        score += 5000
    elif "mixtral" in s:
        score += 4000

    mix_match = re.search(r"(\d+)x(\d+)b", s)
    if mix_match:
        score += int(mix_match.group(1)) * int(mix_match.group(2))
    else:
        size_match = re.search(r"(\d+)b", s)
        if size_match:
            score += int(size_match.group(1))

    if "instruct" in s or "chat" in s or "it" in s:
        score += 50

    return score


def choose_best_model(models: list[dict], prefer_id: str | None = None) -> str | None:
    if prefer_id:
        return prefer_id

    best_id = None
    best_score = -1
    for m in models:
        model_id = m.get("id") or m.get("name")
        if not isinstance(model_id, str):
            continue
        text = f"{model_id} {m.get('name', '')}".lower()
        if any(bad in text for bad in ("embed", "embedding", "rerank", "whisper", "vision")):
            continue
        score = _score_model_id(text)
        if score > best_score:
            best_score = score
            best_id = model_id

    return best_id


def extract_context_length(model: dict | None) -> int | None:
    if not isinstance(model, dict):
        return None
    for key in (
        "context_length",
        "max_context",
        "max_context_length",
        "n_ctx",
        "max_tokens",
    ):
        val = model.get(key)
        if isinstance(val, (int, float)) and val > 0:
            return int(val)
    return None


def openwebui_chat(
    messages: list[dict],
    model_id: str,
    base_url: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    timeout: int = 30,
) -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = requests.post(
        f"{base_url}/api/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def parse_topics_list(value) -> list[int]:
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, str) and value.strip():
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [int(v) for v in parsed]
        except (ValueError, SyntaxError):
            pass
        matches = re.findall(r"-?\d+", value)
        return [int(m) for m in matches]
    return []


def build_prompt_for_hier_topic(
    domain_description: str,
    topic_labels: list[str],
    doc_snippets: list[str],
    max_labels: int,
    doc_count: int,
) -> str:
    labels_block = "\n".join(f"- {lbl}" for lbl in topic_labels) if topic_labels else "(none)"
    docs_block = "\n\n".join(f"Doc {i + 1}:\n{doc}" for i, doc in enumerate(doc_snippets))
    return f"""
You are an expert in {domain_description} and scientometrics.

This is a higher-level (hierarchical) topic that groups several subtopics.
Subtopic labels (from BERTopic): 
{labels_block}

There are {doc_count} documents in this merged topic. Below are representative documents:
{docs_block}

TASK:
1. Propose up to {max_labels} short, human-readable labels for this hierarchical topic.
2. Each label MUST be:
   - a noun phrase (no full sentence)
   - 3 to 6 words long
   - grounded ONLY in the documents and subtopic labels shown above
3. Output labels as a list, one per line. No explanations.
""".strip()


def label_hierarchical_for_cluster(
    cluster: str,
    llm,
    openwebui_config: dict | None,
    output_base_dir: str,
    model_path: str,
    n_ctx: int,
    n_threads: int,
    n_gpu_layers: int,
    max_labels: int,
    doc_sample_fraction: float,
    max_docs_per_node: int,
    prompt_token_budget: int | None,
    max_chars: int,
    top_k_topic_labels: int,
) -> tuple:
    cluster_dir = Path(output_base_dir) / f"cluster_{cluster}"
    hier_path = cluster_dir / "hierarchical_topics_raw.csv"
    if not hier_path.exists():
        print(f"[{cluster}] hierarchical_topics_raw.csv not found at {hier_path}, skipping.")
        return llm, openwebui_config

    docs_path = cluster_dir / "publications_with_topics.csv"
    if not docs_path.exists():
        docs_path = cluster_dir / "docs_with_topics_and_macro_topics.csv"
    if not docs_path.exists():
        print(f"[{cluster}] No document export found, skipping.")
        return llm, openwebui_config

    topic_info_path = cluster_dir / "topic_info.csv"
    topic_info = pd.read_csv(topic_info_path) if topic_info_path.exists() else None

    hier_df = pd.read_csv(hier_path)
    out_path = cluster_dir / "hierarchical_topics_labeled.csv"
    if out_path.exists():
        existing = pd.read_csv(out_path)
        hier_df = hier_df.merge(existing[["Parent_ID", "LLM_Label", "LLM_Labels"]], on="Parent_ID", how="left")
    else:
        hier_df["LLM_Label"] = ""
        hier_df["LLM_Labels"] = ""

    if topic_info is not None and "Topic" in topic_info.columns:
        label_map = {}
        count_map = {}
        for _, row in topic_info.iterrows():
            topic_id = row.get("Topic")
            if pd.isna(topic_id):
                continue
            for col in ("CustomLabel", "CustomName", "Name"):
                value = row.get(col, "")
                if isinstance(value, str) and value.strip():
                    label_map[int(topic_id)] = value.strip()
                    break
            if "Count" in row:
                count_map[int(topic_id)] = row.get("Count")
    else:
        label_map = {}
        count_map = {}

    docs_df = pd.read_csv(docs_path)
    domain_description = CLUSTER_DOMAIN_DESCRIPTION.get(cluster, CLUSTER_DOMAIN_DESCRIPTION["OTHER"])

    def call_llm(messages: list[dict], max_tokens: int, temperature: float) -> str:
        nonlocal llm, openwebui_config
        if openwebui_config and not openwebui_config.get("disabled"):
            retries = int(openwebui_config.get("retries", 3))
            base_sleep = float(openwebui_config.get("retry_sleep", 5))
            for attempt in range(retries):
                try:
                    result = openwebui_chat(
                        messages=messages,
                        model_id=openwebui_config["model_id"],
                        base_url=openwebui_config["base_url"],
                        api_key=openwebui_config["api_key"],
                        max_tokens=max_tokens,
                        temperature=temperature,
                        timeout=openwebui_config.get("timeout", 30),
                    )
                    throttle = float(openwebui_config.get("throttle_sec", 0))
                    if throttle > 0:
                        time.sleep(throttle)
                    return result
                except requests.HTTPError as exc:
                    status = exc.response.status_code if exc.response is not None else None
                    if status == 429 and attempt < retries - 1:
                        sleep_s = base_sleep * (2 ** attempt)
                        print(f"[{cluster}] Rate limit (429). Sleeping {sleep_s:.1f}s and retrying...")
                        time.sleep(sleep_s)
                        continue
                    print(f"[{cluster}] WARNUNG: OpenWebUI fehlgeschlagen ({exc}); falle auf Llama zurück.")
                    openwebui_config["disabled"] = True
                    break
                except Exception as exc:
                    print(f"[{cluster}] WARNUNG: OpenWebUI fehlgeschlagen ({exc}); falle auf Llama zurück.")
                    openwebui_config["disabled"] = True
                    break
        if llm is None:
            if not model_path:
                raise ValueError("Kein Llama-Model-Pfad für Fallback vorhanden.")
            print(f"[{cluster}] Loading Llama model from: {model_path}")
            llm = get_llama_cls()(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
            )
        resp = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp["choices"][0]["message"]["content"]

    print(f"\n=== Cluster {cluster}: labeling hierarchical topics ===")

    for idx, row in hier_df.iterrows():
        if isinstance(row.get("LLM_Label"), str) and row.get("LLM_Label").strip():
            continue
        topics_list = parse_topics_list(row.get("Topics", ""))
        if not topics_list:
            continue

        doc_subset = docs_df[docs_df.get("topic").isin(topics_list)].copy()
        doc_count = len(doc_subset)
        n_sample = max(1, int(round(doc_count * doc_sample_fraction))) if doc_count else 0
        if "topic_prob" in doc_subset.columns:
            doc_subset = doc_subset.sort_values("topic_prob", ascending=False)
        if n_sample:
            doc_subset = doc_subset.head(min(n_sample, doc_count))
        if max_docs_per_node and len(doc_subset) > max_docs_per_node:
            doc_subset = doc_subset.head(max_docs_per_node)

        effective_n_ctx = openwebui_config.get("context_length") if openwebui_config else None
        if not isinstance(effective_n_ctx, int) or effective_n_ctx <= 0:
            effective_n_ctx = n_ctx
        if prompt_token_budget is None:
            budget = max(512, effective_n_ctx - 512)
        else:
            budget = min(prompt_token_budget, max(512, effective_n_ctx - 512))

        doc_snippets = []
        used_tokens = 0
        for _, doc_row in doc_subset.iterrows():
            snippet = format_doc_snippet(
                doc_row.get("title", ""),
                doc_row.get("abstract", ""),
                doc_row.get("keyword", ""),
                max_chars=max_chars,
            )
            snippet_tokens = estimate_tokens(snippet)
            if used_tokens + snippet_tokens > budget:
                break
            doc_snippets.append(snippet)
            used_tokens += snippet_tokens

        topic_labels = []
        if label_map:
            counts = []
            for tid in topics_list:
                lbl = label_map.get(int(tid))
                if lbl:
                    cnt = count_map.get(int(tid), 0)
                    counts.append((lbl, cnt))
            counts.sort(key=lambda x: x[1], reverse=True)
            topic_labels = [c[0] for c in counts[:top_k_topic_labels]]

        prompt = build_prompt_for_hier_topic(
            domain_description=domain_description,
            topic_labels=topic_labels,
            doc_snippets=doc_snippets,
            max_labels=max_labels,
            doc_count=doc_count,
        )

        raw_text = call_llm(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You label research topics for scientific figures. "
                        "Always output short noun phrases."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=64,
            temperature=0.1,
        )

        labels = parse_multi_labels(raw_text, max_labels=max_labels)
        if not labels:
            fallback = topic_labels[0] if topic_labels else f"Hierarchical Topic {row.get('Parent_ID')}"
            labels = [fallback]

        hier_df.loc[idx, "LLM_Label"] = labels[0]
        hier_df.loc[idx, "LLM_Labels"] = "; ".join(labels)
        hier_df.to_csv(out_path, index=False)

    print(f"[{cluster}] Done. Saved labeled hierarchy to: {out_path}")
    return llm, openwebui_config


def parse_args():
    parser = argparse.ArgumentParser(description="Label hierarchical BERTopic topics using an LLM.")
    parser.add_argument("--clusters", default=",".join(CLUSTERS))
    parser.add_argument("--output-base-dir", default=OUTPUT_BASE_DIR)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--max-labels", type=int, default=3)
    parser.add_argument("--doc-sample-fraction", type=float, default=1.0)
    parser.add_argument("--max-docs-per-node", type=int, default=50)
    parser.add_argument("--prompt-token-budget", type=int, default=None)
    parser.add_argument("--max-chars", type=int, default=600)
    parser.add_argument("--top-k-topic-labels", type=int, default=8)
    parser.add_argument("--openwebui-base-url", default=OPENWEBUI_BASE_URL)
    parser.add_argument("--openwebui-model", default=OPENWEBUI_MODEL)
    parser.add_argument("--openwebui-timeout", type=int, default=30)
    parser.add_argument("--openwebui-retries", type=int, default=3)
    parser.add_argument("--openwebui-retry-sleep", type=float, default=5.0)
    parser.add_argument("--openwebui-throttle", type=float, default=0.0)
    parser.add_argument("--use-openwebui", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    clusters = [c.strip() for c in args.clusters.split(",") if c.strip()]

    openwebui_config = None
    if args.use_openwebui or OPENWEBUI_API_KEY:
        if not OPENWEBUI_API_KEY:
            print("OPENWEBUI_API_KEY not set; falling back to local Llama.")
        else:
            try:
                models = list_openwebui_models(
                    base_url=args.openwebui_base_url,
                    api_key=OPENWEBUI_API_KEY,
                    timeout=args.openwebui_timeout,
                )
                model_id = choose_best_model(models, prefer_id=args.openwebui_model)
                if model_id:
                    model_meta = next((m for m in models if m.get("id") == model_id), None)
                    openwebui_config = {
                        "base_url": args.openwebui_base_url,
                        "api_key": OPENWEBUI_API_KEY,
                        "model_id": model_id,
                        "timeout": args.openwebui_timeout,
                        "context_length": extract_context_length(model_meta),
                        "retries": args.openwebui_retries,
                        "retry_sleep": args.openwebui_retry_sleep,
                        "throttle_sec": args.openwebui_throttle,
                        "disabled": False,
                    }
                    print(f"Using OpenWebUI model: {model_id}")
                else:
                    print("No suitable OpenWebUI model found; falling back to local Llama.")
            except Exception as exc:
                print(f"OpenWebUI lookup failed ({exc}); falling back to local Llama.")

    llm = None
    if openwebui_config is None:
        print(f"Loading Llama model from: {args.model_path}")
        llm = get_llama_cls()(
            model_path=args.model_path,
            n_ctx=args.n_ctx,
            n_threads=args.n_threads,
            n_gpu_layers=args.n_gpu_layers,
        )

    for cluster in clusters:
        llm, openwebui_config = label_hierarchical_for_cluster(
            cluster=cluster,
            llm=llm,
            openwebui_config=openwebui_config,
            output_base_dir=args.output_base_dir,
            model_path=args.model_path,
            n_ctx=args.n_ctx,
            n_threads=args.n_threads,
            n_gpu_layers=args.n_gpu_layers,
            max_labels=args.max_labels,
            doc_sample_fraction=args.doc_sample_fraction,
            max_docs_per_node=args.max_docs_per_node,
            prompt_token_budget=args.prompt_token_budget,
            max_chars=args.max_chars,
            top_k_topic_labels=args.top_k_topic_labels,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
