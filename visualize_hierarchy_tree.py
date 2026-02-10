#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import ast
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Create collapsible HTML trees for hierarchical topics.")
    parser.add_argument("--clusters", default="ALL,CS,SS,BIO,OTHER")
    parser.add_argument("--output-base-dir", default="outputs_by_cluster")
    parser.add_argument("--output-name", default="hierarchical_topics_tree.html")
    parser.add_argument("--collapsed-depth", type=int, default=1)
    parser.add_argument("--max-label-length", type=int, default=120)
    parser.add_argument("--node-gap-x", type=int, default=180, help="Horizontal spacing between nodes")
    parser.add_argument("--node-gap-y", type=int, default=160, help="Vertical spacing between levels")
    parser.add_argument("--width", type=int, default=1800, help="SVG width")
    parser.add_argument("--wrap-width", type=int, default=180, help="Max label width in pixels")
    parser.add_argument("--line-height", type=int, default=14, help="Line height for wrapped labels")
    return parser.parse_args()


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
    return []


def load_topic_labels(cluster_dir: Path) -> tuple[dict[int, str], dict[int, int], dict[int, str]]:
    labeled = cluster_dir / "topic_info_labeled.csv"
    unlabeled = cluster_dir / "topic_info.csv"
    if labeled.exists():
        df = pd.read_csv(labeled)
        label_cols = ["LLM_Label", "CustomLabel", "CustomName", "Name"]
    elif unlabeled.exists():
        df = pd.read_csv(unlabeled)
        label_cols = ["CustomLabel", "CustomName", "Name"]
    else:
        return {}, {}, {}

    label_map = {}
    count_map = {}
    llm_labels_map = {}
    for _, row in df.iterrows():
        topic_id = row.get("Topic")
        if pd.isna(topic_id) or int(topic_id) == -1:
            continue
        for col in label_cols:
            value = row.get(col, "")
            if isinstance(value, str) and value.strip():
                label_map[int(topic_id)] = value.strip()
                break
        llm_all = row.get("LLM_Labels", "")
        if isinstance(llm_all, str) and llm_all.strip():
            llm_labels_map[int(topic_id)] = llm_all.strip()
        if "Count" in row and not pd.isna(row.get("Count")):
            count_map[int(topic_id)] = int(row.get("Count"))
    return label_map, count_map, llm_labels_map


def load_hier_labels(cluster_dir: Path) -> dict[int, dict[str, str]]:
    labeled = cluster_dir / "hierarchical_topics_labeled.csv"
    if not labeled.exists():
        return {}
    df = pd.read_csv(labeled)
    label_map = {}
    for _, row in df.iterrows():
        parent_id = row.get("Parent_ID")
        if pd.isna(parent_id):
            continue
        label = row.get("LLM_Label", "")
        llm_all = row.get("LLM_Labels", "")
        if isinstance(label, str) and label.strip():
            label_map[int(parent_id)] = {
                "primary": label.strip(),
                "all": llm_all.strip() if isinstance(llm_all, str) else "",
            }
    return label_map


def build_tree(
    hier_df: pd.DataFrame,
    topic_labels: dict[int, str],
    topic_counts: dict[int, int],
    hier_labels: dict[int, dict[str, str]],
    max_label_length: int,
    wrap_chars: int,
    topic_llm_labels: dict[int, str],
):
    parent_rows = {}
    parent_ids = set()
    child_ids = set()

    for _, row in hier_df.iterrows():
        parent_id = int(row["Parent_ID"])
        left_id = int(row["Child_Left_ID"])
        right_id = int(row["Child_Right_ID"])
        parent_rows[parent_id] = row
        parent_ids.add(parent_id)
        child_ids.add(left_id)
        child_ids.add(right_id)

    root_ids = list(parent_ids - child_ids)
    if not root_ids:
        root_ids = list(parent_ids)

    def clip(text: str) -> str:
        if not isinstance(text, str):
            return ""
        if max_label_length and len(text) > max_label_length:
            return text[: max_label_length - 3].rstrip() + "..."
        return text

    def line_count(text: str) -> int:
        if not text:
            return 1
        words = text.split()
        if not words:
            return 1
        lines = 1
        current = 0
        for word in words:
            wlen = len(word)
            if current == 0:
                current = wlen
            elif current + 1 + wlen <= wrap_chars:
                current += 1 + wlen
            else:
                lines += 1
                current = wlen
        return max(1, lines)

    def node_label(node_id: int) -> tuple[str, int | None, str]:
        label_entry = hier_labels.get(node_id) or {}
        label = label_entry.get("primary", "") if isinstance(label_entry, dict) else ""
        llm_all = label_entry.get("all", "") if isinstance(label_entry, dict) else ""
        count = None
        if label:
            row = parent_rows.get(node_id)
            topics_list = parse_topics_list(row.get("Topics", "")) if row is not None else []
            if topics_list:
                count = sum(topic_counts.get(t, 0) for t in topics_list)
        return clip(label), count, llm_all

    def leaf_label(node_id: int) -> tuple[str, int | None, str]:
        label = topic_labels.get(node_id, f"Topic {node_id}")
        count = topic_counts.get(node_id)
        llm_all = topic_llm_labels.get(node_id, "")
        return clip(label), count, llm_all

    def build_node(node_id: int) -> dict:
        if node_id in parent_rows:
            row = parent_rows[node_id]
            left_id = int(row["Child_Left_ID"])
            right_id = int(row["Child_Right_ID"])
            label, count, llm_all = node_label(node_id)
            if not label:
                label = row.get("Parent_Name", f"Parent {node_id}")
            name = label
            if count is not None:
                name = f"{name} (n={count})"
            return {
                "id": f"P{node_id}",
                "name": name,
                "lines": line_count(name),
                "tooltip": llm_all,
                "children": [build_node(left_id), build_node(right_id)],
            }
        label, count, llm_all = leaf_label(node_id)
        name = f"{label} (T{node_id})"
        if count is not None:
            name = f"{name}, n={count}"
        return {
            "id": f"T{node_id}",
            "name": name,
            "lines": line_count(name),
            "tooltip": llm_all,
        }

    if len(root_ids) == 1:
        return build_node(root_ids[0])
    return {"id": "ROOT", "name": "All Topics", "children": [build_node(rid) for rid in root_ids]}


def render_html(
    tree_data: dict,
    collapsed_depth: int,
    dx: int,
    dy: int,
    width: int,
    wrap_width: int,
    line_height: int,
) -> str:
    data_json = json.dumps(tree_data)
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Hierarchical Topics (LLM)</title>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <style>
    body {
      font-family: Arial, sans-serif;
      margin: 0;
      padding: 16px;
      background: #fafafa;
    }
    #chart {
      border: 1px solid #e5e7eb;
      background: #ffffff;
      max-height: 90vh;
      overflow: auto;
    }
    .node circle {
      fill: #4c78a8;
      stroke: #2b4c6f;
      stroke-width: 1px;
    }
    .node text {
      font-size: 12px;
      fill: #1f2933;
    }
    .node .toggle {
      font-size: 12px;
      fill: #fff;
      pointer-events: none;
    }
    .link {
      fill: none;
      stroke: #9fb3c8;
      stroke-width: 1.2px;
    }
  </style>
</head>
<body>
<h2>Hierarchical Topics (LLM Labels)</h2>
<div id="chart"></div>
<script>
const treeData = __DATA_JSON__;
const collapsedDepth = __COLLAPSED_DEPTH__;

const width = __WIDTH__;
const dx = __DX__;
const dy = __DY__;
const margin = {top: 20, left: 60};
const wrapWidth = __WRAP_WIDTH__;
const lineHeight = __LINE_HEIGHT__;

const root = d3.hierarchy(treeData);
root.x0 = 0;
root.y0 = 0;

function collapse(d) {
  if (d.children) {
    d._children = d.children;
    d._children.forEach(collapse);
    d.children = null;
  }
}

root.descendants().forEach(d => {
  if (d.depth >= collapsedDepth) {
    collapse(d);
  }
});

const tree = d3.tree()
  .nodeSize([dx, dy])
  .separation((a, b) => {
    const la = a.data.lines || 1;
    const lb = b.data.lines || 1;
    const base = a.parent === b.parent ? 1 : 1.2;
    return base * Math.max(la, lb);
  });
const diagonal = d3.linkVertical().x(d => d.x).y(d => d.y);

const svg = d3.select("#chart").append("svg")
  .attr("width", width)
  .style("font", "12px sans-serif");

const gMain = svg.append("g");
const gLink = gMain.append("g").attr("class", "links");
const gNode = gMain.append("g").attr("class", "nodes");

function update(source) {
  const sourcePos = {x: (source.x0 ?? source.x), y: (source.y0 ?? source.y)};
  const nodes = root.descendants();
  const links = root.links();

  tree(root);

  let left = root;
  let right = root;
  root.eachBefore(node => {
    if (node.x < left.x) left = node;
    if (node.x > right.x) right = node;
  });

  const widthNeeded = right.x - left.x + dx * 4;
  const heightNeeded = root.height * dy + dy * 2;
  const svgWidth = Math.max(width, widthNeeded) + margin.left + 40;
  const svgHeight = heightNeeded + margin.top + 40;
  svg.attr("width", svgWidth).attr("height", svgHeight);
  gMain.attr("transform", `translate(${margin.left - left.x + dx * 2},${margin.top})`);

  const node = gNode.selectAll("g").data(nodes, d => d.id);

  const nodeEnter = node.enter().append("g")
    .attr("class", "node")
    .attr("transform", d => `translate(${sourcePos.x},${sourcePos.y})`)
    .on("click", (event, d) => {
      if (d.children) {
        d._children = d.children;
        d.children = null;
      } else {
        d.children = d._children;
        d._children = null;
      }
      update(d);
    });

  nodeEnter.append("circle")
    .attr("r", 5);

  const label = nodeEnter.append("text")
    .attr("dy", "0.31em")
    .attr("x", d => d.children || d._children ? -10 : 10)
    .attr("text-anchor", d => d.children || d._children ? "end" : "start")
    .text(d => d.data.name);

  label.call(wrap, wrapWidth);
  label.clone(true).lower().attr("stroke", "white");

  nodeEnter.append("text")
    .attr("class", "toggle")
    .attr("dy", "0.31em")
    .attr("x", -2)
    .attr("text-anchor", "middle")
    .text(d => d._children ? "+" : d.children ? "-" : "");

  nodeEnter.append("title").text(d => d.data.tooltip || d.data.name);

  const nodeUpdate = nodeEnter.merge(node);
  nodeUpdate.transition().duration(250)
    .attr("transform", d => `translate(${d.x},${d.y})`);

  nodeUpdate.select("text.toggle")
    .text(d => d._children ? "+" : d.children ? "-" : "");

  const nodeExit = node.exit().transition().duration(250)
    .attr("transform", d => `translate(${sourcePos.x},${sourcePos.y})`)
    .remove();

  nodeExit.select("circle").attr("r", 0);

  const link = gLink.selectAll("path").data(links, d => d.target.id);

  const linkEnter = link.enter().append("path")
    .attr("class", "link")
    .attr("d", d => {
      const o = {x: sourcePos.x, y: sourcePos.y};
      return diagonal({source: o, target: o});
    });

  linkEnter.merge(link).transition().duration(250)
    .attr("d", diagonal);

  link.exit().transition().duration(250)
    .attr("d", d => {
      const o = {x: sourcePos.x, y: sourcePos.y};
      return diagonal({source: o, target: o});
    })
    .remove();

  root.eachBefore(d => {
    d.x0 = d.x;
    d.y0 = d.y;
  });
}

update(root);

function wrap(text, width) {
  text.each(function() {
    const text = d3.select(this);
    const words = text.text().split(/\\s+/).reverse();
    let word;
    let line = [];
    let lineNumber = 0;
    const x = text.attr("x");
    const y = text.attr("y");
    const dyText = parseFloat(text.attr("dy")) || 0;
    let tspan = text.text(null).append("tspan")
      .attr("x", x)
      .attr("y", y)
      .attr("dy", dyText + "em");

    while ((word = words.pop())) {
      line.push(word);
      tspan.text(line.join(" "));
      if (tspan.node().getComputedTextLength() > width) {
        line.pop();
        tspan.text(line.join(" "));
        line = [word];
        tspan = text.append("tspan")
          .attr("x", x)
          .attr("y", y)
          .attr("dy", (lineNumber + 1) * (lineHeight / 12) + "em")
          .text(word);
        lineNumber += 1;
      }
    }
  });
}
</script>
</body>
</html>
"""
    return (
        template.replace("__DATA_JSON__", data_json)
        .replace("__COLLAPSED_DEPTH__", str(collapsed_depth))
        .replace("__DX__", str(dx))
        .replace("__DY__", str(dy))
        .replace("__WIDTH__", str(width))
        .replace("__WRAP_WIDTH__", str(wrap_width))
        .replace("__LINE_HEIGHT__", str(line_height))
    )


def main():
    args = parse_args()
    clusters = [c.strip() for c in args.clusters.split(",") if c.strip()]

    for cluster in clusters:
        cluster_dir = Path(args.output_base_dir) / f"cluster_{cluster}"
        hier_path = cluster_dir / "hierarchical_topics_raw.csv"
        if not hier_path.exists():
            print(f"[{cluster}] hierarchical_topics_raw.csv not found, skipping.")
            continue

        hier_df = pd.read_csv(hier_path)
        topic_labels, topic_counts, topic_llm_labels = load_topic_labels(cluster_dir)
        hier_labels = load_hier_labels(cluster_dir)

        wrap_chars = max(10, int(args.wrap_width // 7)) if args.wrap_width else 40
        tree_data = build_tree(
            hier_df,
            topic_labels,
            topic_counts,
            hier_labels,
            max_label_length=args.max_label_length,
            wrap_chars=wrap_chars,
            topic_llm_labels=topic_llm_labels,
        )
        html = render_html(
            tree_data,
            collapsed_depth=args.collapsed_depth,
            dx=args.node_gap_x,
            dy=args.node_gap_y,
            width=args.width,
            wrap_width=args.wrap_width,
            line_height=args.line_height,
        )

        out_path = cluster_dir / args.output_name
        out_path.write_text(html, encoding="utf-8")
        print(f"[{cluster}] Wrote {out_path}")


if __name__ == "__main__":
    main()
