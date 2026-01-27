#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate descriptive stats for the abstracts dataset."
    )
    parser.add_argument("--data-path", default="abstracts.csv")
    parser.add_argument("--sep", default=r"\$\$")
    parser.add_argument("--output-dir", default="outputs_summary")
    parser.add_argument("--keyword-column", default="keyword")
    parser.add_argument("--title-column", default="title")
    parser.add_argument("--abstract-column", default="abstract")
    parser.add_argument("--year-column", default="year")
    parser.add_argument("--source-column", default="source")
    parser.add_argument("--middle-group-column", default="middle_group")
    parser.add_argument("--subject-area-column", default="subject_area")
    parser.add_argument("--top-keywords", type=int, default=50)
    return parser.parse_args()


def map_middle_group_to_cluster(mg: str) -> str:
    if not isinstance(mg, str):
        return "OTHER"
    mg = mg.strip()

    ss_set = {
        "Social Sciences",
        "Psychology",
        "Economics, Econometrics and Finance",
        "Business, Management and Accounting",
        "Arts and Humanities",
        "Education",
    }

    cs_set = {
        "Computer Science",
        "Mathematics",
        "Decision Sciences",
        "Engineering",
    }

    bio_set = {
        "Medicine",
        "Neuroscience",
        "Biochemistry, Genetics and Molecular Biology",
        "Health Professions",
        "Immunology and Microbiology",
    }

    if mg in ss_set:
        return "SS"
    if mg in cs_set:
        return "CS"
    if mg in bio_set:
        return "BIO"
    return "OTHER"


def classify_document(middle_group_value: str, abstract: str) -> str:
    if not isinstance(middle_group_value, str):
        middle_group_value = ""
    if not isinstance(abstract, str):
        abstract = ""

    groups = [g.strip() for g in re.split(r"[;|,]", middle_group_value) if g.strip()]
    abs_low = abstract.lower()

    ss_keywords = [
        "attitudes",
        "perceptions",
        "media",
        "voters",
        "election",
        "democracy",
        "public opinion",
        "cohort",
        "inequality",
        "trust",
        "respondents",
        "survey",
        "panel",
        "questionnaire",
        "social sciences",
        "political",
    ]
    cs_keywords = [
        "language model",
        "large language model",
        "llm",
        "bert",
        "embedding",
        "neural",
        "deep learning",
        "machine learning",
        "transformer",
        "algorithm",
        "classification",
        "retrieval",
        "entity extraction",
        "information retrieval",
        "nlp",
        "corpus",
    ]

    mapped = [map_middle_group_to_cluster(g) for g in groups]

    if "SS" in mapped:
        return "SS"
    if "CS" in mapped:
        if any(k in abs_low for k in ss_keywords):
            return "SS"
        return "CS"
    if "BIO" in mapped:
        return "BIO"

    if any(k in abs_low for k in cs_keywords):
        return "CS"
    if any(k in abs_low for k in ss_keywords):
        return "SS"

    return "OTHER"


def split_keywords(raw: str) -> list[str]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    parts = re.split(r"[;|,]", raw)
    return [p.strip().lower() for p in parts if p.strip()]


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_path, sep=args.sep, engine="python", header=0)

    required = {
        args.title_column,
        args.abstract_column,
        args.year_column,
        args.source_column,
        args.middle_group_column,
    }
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    summary = {
        "total_documents": int(len(df)),
    }

    def safe_counts(series: pd.Series) -> pd.DataFrame:
        return series.fillna("UNKNOWN").value_counts().reset_index().rename(
            columns={"index": "value", series.name: "count"}
        )

    # Cluster distribution (using same rules as modeling)
    df["cluster"] = df.apply(
        lambda row: classify_document(
            row.get(args.middle_group_column, ""),
            row.get(args.abstract_column, ""),
        ),
        axis=1,
    )
    safe_counts(df["cluster"]).to_csv(output_dir / "counts_by_cluster.csv", index=False)

    # Middle group distribution
    safe_counts(df[args.middle_group_column]).to_csv(
        output_dir / "counts_by_middle_group.csv", index=False
    )

    # Subject area distribution (if present)
    if args.subject_area_column in df.columns:
        safe_counts(df[args.subject_area_column]).to_csv(
            output_dir / "counts_by_subject_area.csv", index=False
        )

    # Source distribution
    safe_counts(df[args.source_column]).to_csv(
        output_dir / "counts_by_source.csv", index=False
    )

    # Year distribution
    df_year = pd.to_numeric(df[args.year_column], errors="coerce")
    df_year = df_year.dropna().astype(int)
    df_year.value_counts().sort_index().reset_index().rename(
        columns={"index": "year", args.year_column: "count"}
    ).to_csv(output_dir / "counts_by_year.csv", index=False)

    # Missingness summary
    missing = df.isna().sum().sort_values(ascending=False).to_frame("missing_count")
    missing.to_csv(output_dir / "missingness_by_column.csv")

    # Top keywords
    if args.keyword_column in df.columns:
        counter = Counter()
        for raw in df[args.keyword_column].tolist():
            counter.update(split_keywords(raw))
        top_k = counter.most_common(args.top_keywords)
        pd.DataFrame(top_k, columns=["keyword", "count"]).to_csv(
            output_dir / "top_keywords.csv", index=False
        )

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"Done. Wrote outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
