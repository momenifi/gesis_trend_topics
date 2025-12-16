# GESIS Trend Topics

This repository runs the BERTopic pipeline grouped by discipline clusters (SS, CS, BIO, OTHER).

## How to run

1. Install Python dependencies (for example via `pip install -r requirements.txt` if you have one, or install `bertopic`, `pandas`, `numpy`, `umap-learn`, `hdbscan`, and `pyyaml`).
2. Ensure your input CSV is available. By default the script looks for `abstracts.csv` in the repository root. The CSV must be separated by `$$` and contain the columns `title`, `abstract`, `year`, `source`, and `middle_group`.
3. Run the pipeline:
   ```bash
   python hierarchical_topics_disiplines.py
   ```
4. To override defaults, provide a YAML config:
   ```bash
   python hierarchical_topics_disiplines.py --config config.example.yaml
   ```

The outputs are written under `outputs_by_cluster/cluster_<LABEL>/` for each cluster.

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
