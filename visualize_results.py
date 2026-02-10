#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Create slide-ready charts for topics.")
    parser.add_argument("--cluster", default="ALL")
    parser.add_argument("--output-base-dir", default="outputs_by_cluster")
    parser.add_argument("--output-dir", default="outputs_viz")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--year-start", type=int, default=2023)
    parser.add_argument("--year-end", type=int, default=2025)
    return parser.parse_args()


def load_topic_labels(cluster_dir: Path) -> pd.DataFrame:
    labeled = cluster_dir / "topic_info_labeled.csv"
    unlabeled = cluster_dir / "topic_info.csv"
    if labeled.exists():
        df = pd.read_csv(labeled)
    elif unlabeled.exists():
        df = pd.read_csv(unlabeled)
    else:
        raise FileNotFoundError(
            f"Missing topic info in {cluster_dir}. Expected topic_info_labeled.csv or topic_info.csv."
        )

    if "LLM_Label" not in df.columns:
        df["LLM_Label"] = ""

    def pick_label(row: pd.Series) -> str:
        for col in ("LLM_Label", "CustomLabel", "CustomName", "Name"):
            value = row.get(col, "")
            if isinstance(value, str) and value.strip():
                return value.strip()
        return f"Topic {row.get('Topic')}"

    df["Label"] = df.apply(pick_label, axis=1)
    return df


def prepare_top_topics_table(topic_info: pd.DataFrame, top_n: int) -> pd.DataFrame:
    df = topic_info[topic_info["Topic"] != -1].copy()
    df = df.sort_values("Count", ascending=False).head(top_n)
    return df[["Topic", "Label", "Count"]]


def prepare_before_after_table(
    raw_info: pd.DataFrame, labeled_info: pd.DataFrame, top_n: int
) -> pd.DataFrame:
    labeled = labeled_info[labeled_info["Topic"] != -1].copy()
    labeled = labeled.sort_values("Count", ascending=False).head(top_n)
    raw = raw_info[raw_info["Topic"] != -1].copy()
    raw_col = "Name" if "Name" in raw.columns else "Representation"
    raw = raw[["Topic", raw_col]].rename(columns={raw_col: "Raw Terms"})
    merged = labeled.merge(raw, on="Topic", how="left")
    return merged[["Raw Terms", "Label", "Count"]].rename(
        columns={"Label": "LLM Label"}
    )


def prepare_topics_over_time(
    cluster_dir: Path,
    labels_df: pd.DataFrame,
    year_start: int,
    year_end: int,
    top_n: int,
) -> pd.DataFrame:
    tot_path = cluster_dir / "topics_over_time.csv"
    if not tot_path.exists():
        raise FileNotFoundError(f"Missing topics_over_time.csv at {tot_path}.")

    tot = pd.read_csv(tot_path)
    tot = tot[tot["Topic"] != -1].copy()
    tot = tot.rename(columns={"Timestamp": "Year"})
    tot["Year"] = pd.to_numeric(tot["Year"], errors="coerce").astype("Int64")
    tot = tot[(tot["Year"] >= year_start) & (tot["Year"] <= year_end)]

    totals = (
        tot.groupby("Topic", as_index=False)["Frequency"]
        .sum()
        .sort_values("Frequency", ascending=False)
        .head(top_n)
    )
    top_topics = set(totals["Topic"].tolist())

    label_map = dict(zip(labels_df["Topic"], labels_df["Label"]))
    tot = tot[tot["Topic"].isin(top_topics)].copy()
    tot["Label"] = tot["Topic"].map(label_map)

    labeled_long = tot[["Year", "Topic", "Label", "Frequency"]].copy()

    pivot = tot.pivot_table(
        index="Year",
        columns="Label",
        values="Frequency",
        fill_value=0,
        aggfunc="sum",
    ).sort_index()

    return pivot, labeled_long


def plot_table_png(df: pd.DataFrame, path: Path, title: str):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 0.6 * (len(df) + 2)))
    ax.axis("off")
    col_widths = None
    if list(df.columns) == ["Topic", "Label", "Count"]:
        col_widths = [0.12, 0.68, 0.2]
    elif list(df.columns) == ["Topic", "Raw Terms", "Count"]:
        col_widths = [0.12, 0.68, 0.2]
    elif list(df.columns) == ["Raw Terms", "LLM Label", "Count"]:
        col_widths = [0.58, 0.36, 0.06]
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="left",
        loc="center",
        colWidths=col_widths,
    )
    for cell in table.get_celld().values():
        cell.PAD = 0.02
    if list(df.columns) in (
        ["Topic", "Label", "Count"],
        ["Topic", "Raw Terms", "Count"],
        ["Raw Terms", "LLM Label", "Count"],
    ):
        for (row, col), cell in table.get_celld().items():
            if col in (0, 1, 2):
                cell.get_text().set_ha("center")
            if row > 0 and col in (0, 1):
                cell.get_text().set_fontweight("bold")
            if row > 0 and col == 0:
                cell.get_text().set_color("#1F4E79")
                cell.get_text().set_ha("left")
            if row > 0 and col == 1:
                cell.get_text().set_color("#7A2E2E")
                cell.get_text().set_ha("left")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    # Header styling
    for col in range(len(df.columns)):
        header_cell = table[(0, col)]
        header_cell.set_facecolor("#2F4B7C")
        header_cell.get_text().set_color("white")
        header_cell.get_text().set_fontweight("bold")
    # Alternating row colors (skip header row)
    for row in range(1, len(df) + 1):
        row_color = "#F2F2F2" if row % 2 == 0 else "white"
        for col in range(len(df.columns)):
            table[(row, col)].set_facecolor(row_color)
    ax.set_title(title, pad=4)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_trends_png(pivot: pd.DataFrame, path: Path, title: str):
    import matplotlib.pyplot as plt

    def build_color_cycle(n: int):
        if n <= 0:
            return []
        cmap = plt.get_cmap("tab20")
        if n <= cmap.N:
            return [cmap(i) for i in range(n)]
        return [plt.cm.hsv(i / n) for i in range(n)]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = build_color_cycle(len(pivot.columns))
    for label, color in zip(pivot.columns, colors):
        ax.plot(pivot.index, pivot[label], marker="o", label=label, color=color)
    fig.suptitle(title, y=0.98)
    ax.set_xlabel("Year")
    ax.set_ylabel("Frequency")
    ax.set_xticks(sorted(pivot.index.unique()))
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=8,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _split_rect(rect, sizes):
    if not sizes:
        return []
    if len(sizes) == 1:
        return [rect]
    x, y, w, h = rect
    total = sum(sizes)
    first = sizes[0]
    if w >= h:
        w1 = w * (first / total)
        rect1 = (x, y, w1, h)
        rect2 = (x + w1, y, w - w1, h)
    else:
        h1 = h * (first / total)
        rect1 = (x, y, w, h1)
        rect2 = (x, y + h1, w, h - h1)
    return [rect1] + _split_rect(rect2, sizes[1:])


def plot_treemap_png(df: pd.DataFrame, path: Path, title: str):
    import matplotlib.pyplot as plt

    labels = df["Label"].tolist()
    sizes = df["Count"].tolist()
    sizes = [s if s > 0 else 1 for s in sizes]
    min_size = min(sizes) if sizes else 1
    max_size = max(sizes) if sizes else 1
    rects = _split_rect((0, 0, 1, 1), sizes)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    cmap = plt.get_cmap("Pastel1")
    for i, (label, size, rect) in enumerate(zip(labels, sizes, rects)):
        x, y, w, h = rect
        color = cmap(i % 9)
        ax.add_patch(
            plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="white", alpha=0.85)
        )
        area = w * h
        if max_size > min_size:
            font_size = 8 + (size - min_size) * (14 - 8) / (max_size - min_size)
        else:
            font_size = 10
        if area > 0.03:
            ax.text(
                x + w * 0.02,
                y + h * 0.5,
                f"{label} ({size})",
                va="center",
                ha="left",
                fontsize=font_size,
                fontweight="bold",
                color="black",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5),
            )

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    cluster_dir = Path(args.output_base_dir) / f"cluster_{args.cluster}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_df = load_topic_labels(cluster_dir)

    top_table = prepare_top_topics_table(labels_df, args.top_n)

    raw_info_path = cluster_dir / "topic_info.csv"
    if raw_info_path.exists():
        raw_info = pd.read_csv(raw_info_path)
        before_after = prepare_before_after_table(raw_info, labels_df, args.top_n)
        plot_table_png(
            before_after,
            output_dir / f"top_topics_{args.cluster}.png",
            f"Top {args.top_n} Topics (Raw vs LLM) - {args.cluster}",
        )
    else:
        plot_table_png(
            top_table,
            output_dir / f"top_topics_{args.cluster}.png",
            f"Top {args.top_n} Topics - {args.cluster}",
        )
    plot_treemap_png(
        top_table,
        output_dir / f"treemap_topics_{args.cluster}.png",
        f"Top {args.top_n} Topics (Treemap) - {args.cluster}",
    )

    pivot, labeled_long = prepare_topics_over_time(
        cluster_dir,
        labels_df,
        args.year_start,
        args.year_end,
        args.top_n,
    )
    # CSV outputs removed; slide-ready images only

    plot_trends_png(
        pivot,
        output_dir / f"topic_trends_{args.cluster}.png",
        f"Topic Trends {args.year_start}-{args.year_end} - {args.cluster}",
    )

    print(f"Done. Wrote charts to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
