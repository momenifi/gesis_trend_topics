# GESIS Trend Topics

This repository runs the BERTopic pipeline grouped by discipline clusters (SS, CS, BIO, OTHER) and once over all disciplines combined.

## How to run

1. Install Python dependencies (recommended via `pip install -r requirements.txt`; includes `bertopic`, `pandas`, `numpy`, `umap-learn`, `hdbscan`, `pyyaml`, and `scikit-learn`).
2. Ensure your input CSV is available. By default the script looks for `abstracts.csv` in the repository root. The CSV must be separated by `$$` and contain the columns `title`, `abstract`, `year`, `source`, and `middle_group`.
3. Run the pipeline:
   ```bash
   python hierarchical_topics_disiplines.py
   ```
4. To override defaults, provide a YAML config:
   ```bash
   python hierarchical_topics_disiplines.py --config config.example.yaml
   ```

The outputs are written under `outputs_by_cluster/cluster_<LABEL>/` for each cluster. A combined run over all documents is stored in `outputs_by_cluster/cluster_ALL/`.

## Labeling with LLM

Use the LLM labeling script after topic modeling. If `OPENWEBUI_API_KEY` is set, it will select the best available OpenWebUI model (unless `OPENWEBUI_MODEL` or `--openwebui-model` is provided). If not, it falls back to local Llama.

Environment variables (optional):
- `OPENWEBUI_API_KEY`
- `OPENWEBUI_MODEL` (explicit model id)
- `OPENWEBUI_BASE_URL` (default: `https://ai-openwebui.gesis.org`)

Example parameters used in this project:

```bash
python label_topics_with_llama_all.py --clusters ALL,CS,SS,BIO,OTHER --n-ctx 8192 --prompt-token-budget 7500
```

To label hierarchical topics (using abstracts + keywords from the merged subtopics), run:

```bash
python label_hierarchical_topics_with_llm.py --clusters ALL,CS,SS,BIO,OTHER --use-openwebui
```

To visualize hierarchical labels as a collapsible tree (plus/minus):

```bash
python visualize_hierarchy_tree.py --clusters ALL,CS,SS,BIO,OTHER
```

In the tree, the primary label is shown and additional LLM labels appear on hover.

To show raw/BERTopic labels instead of LLM labels:

```bash
python visualize_hierarchy_tree.py --clusters ALL,CS,SS,BIO,OTHER --label-source raw
```

To generate both versions in one run (outputs `*_llm.html` and `*_raw.html`):

```bash
python visualize_hierarchy_tree.py --clusters ALL,CS,SS,BIO,OTHER --label-source both
```

## Report

A non-technical summary of the topic modeling and LLM labeling process is available in `REPORT.md`.

## Configuration

You can override runtime settings via a YAML mapping. A ready-to-copy template lives at `config.example.yaml`:

```yaml
data_path: abstracts.csv        # Path to the $$-separated CSV file
output_base_dir: outputs_by_cluster  # Where to write results
embedding_model: paraphrase-multilingual-MiniLM-L12-v2
min_cluster_size: 10
n_components: 5
n_neighbors: 15
random_state: 42
min_docs_for_cluster: 10
```

Pass the path to your customized file with `--config /path/to/your_config.yaml`.

## How to commit and push changes

If you have Git access and want to share your updates:

1. Check what changed:
   ```bash
   git status
   ```
2. Stage and commit your edits with a descriptive message:
   ```bash
   git add .
   git commit -m "Describe your change"
   ```
3. Push to the current branch (replace `origin` or branch name if different):
   ```bash
   git push origin $(git branch --show-current)
   ```
After pushing, open a pull request in your Git hosting service if needed.
