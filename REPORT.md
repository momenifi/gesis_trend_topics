# Topic Modeling Report 

## Goal
We wanted to discover the main research topics in our publication corpus, track how these topics change over time, and provide human-readable labels that make the topics easy to interpret.

## Data Used
Each publication includes:
- Title
- Abstract
- Keywords
- Year
- Source / Discipline information

We grouped publications into four broad discipline clusters: Social Sciences (SS), Computer Science (CS), Biomedical/Health (BIO), and Other (OTHER). We also ran an "ALL" cluster that combines all disciplines.

## How BERTopic Works (in simple terms)
BERTopic works in three stages:

1) Understand each document's meaning
Instead of only counting words, BERTopic converts each abstract into a semantic "embedding" (a numerical representation of meaning).

2) Group similar documents into topics
Documents with similar meaning are clustered together. Each cluster becomes a topic.

3) Summarize each topic with key terms
BERTopic extracts representative keywords that summarize each topic.

Model used for document understanding:
- paraphrase-multilingual-MiniLM-L12-v2 (multilingual sentence-embedding model)

## Our Topic Modeling Process
1) Data preparation
- Cleaned text (titles, abstracts, keywords)
- Combined these fields to represent each publication

2) Discipline clustering
Each publication is assigned to SS, CS, BIO, or OTHER based on its discipline metadata.

3) Topic modeling per cluster
BERTopic was run separately for each cluster so topics are more coherent and discipline-specific.

4) Trends over time
We computed how often each topic appears by year to see which topics are growing or declining.

## How Publications Get Assigned to Topics
Each publication is assigned to the most relevant topic by BERTopic. We also compute a topic probability score, which shows how confident the model is about the assignment.

## Example: Publication Assigned to a Topic
Example entry from `outputs_by_cluster/cluster_SS/publications_with_topics.csv` (fields abbreviated):

```
Title: Fixed is not the opposite of growth: Item keying matters for measuring mindsets
Cluster: SS
Topic: 0
TopicLabel (BERTopic): germany, migration, differences, personality, election
TopicProb: 0.1926
```

## How We Label Topics (LLM-Based)
BERTopic's keywords are useful but sometimes vague. To make topics easier to interpret, we label them using an LLM.

LLM used:
- OpenWebUI (best available model from your account; can be overridden by `OPENWEBUI_MODEL`)
- Fallback: local Llama if OpenWebUI is not available

How labeling works:
- For each topic, we sort publications by the topic probability score (highest first).
- We then feed as many publications as fit in the model context window (title + abstract + keywords), starting from the highest-confidence ones.
- The LLM proposes up to 3 short, human-readable labels grounded in those documents.
- Labels are short noun phrases and must match the topic's documents.

This approach ties labels directly to real publications, which makes them more trustworthy than labels based only on keywords.

## How We Label Hierarchical Topics (LLM-Based)
Hierarchical topics are higher-level merges of multiple leaf topics. We label them by:
- Collecting the documents from all child topics
- Sampling the highest-confidence documents (title + abstract + keywords)
- Passing these to the LLM to produce up to 3 labels

This produces broader labels that describe the overall theme of a hierarchy node.

Notes:
- Each hierarchy node has a primary label (`LLM_Label`) and optional additional labels (`LLM_Labels`).
- In the collapsible tree visualization (`hierarchical_topics_tree.html`), the primary label is shown in the node and additional labels appear on mouse hover.

## Stopwords (How We Handle Them)
Stopwords are common words that don’t help distinguish topics (e.g., “study”, “results”, “data”). We remove them to improve topic quality.

We use two sources of stopwords:
1) Standard English stopwords (from scikit-learn’s `ENGLISH_STOP_WORDS`).
2) A custom list of research-generic terms such as “research”, “study”, “analysis”, “data”, “method”, and “results”.

Additionally, for the CS cluster we remove ML/NLP-specific boilerplate terms (e.g., “model”, “learning”, “embedding”, “transformer”, “bert”, “gpt”), so topics are not dominated by generic ML vocabulary.

These stopwords are applied in the `CountVectorizer` before topic terms are extracted, which helps surface more informative, discipline-specific keywords.

## Outputs Produced
For each discipline cluster (e.g., `outputs_by_cluster/cluster_SS/`):
- `topic_info_raw.csv` contains the raw topic terms before any labeling.
- `topic_info.csv` contains BERTopic’s own automatic labels. These appear in `Name`, `CustomName`, and `CustomLabel` (often identical because they come from the same BERTopic label generation step).
- `topic_info_labeled.csv` adds LLM-generated labels in `LLM_Label` (first label) and `LLM_Labels` (all labels). The BERTopic label columns are still present there, so the file has both sources side by side.
- `topics_over_time.csv` contains topic frequencies by year.
- `topics_over_time_labeled.csv` is the same as above, but with LLM labels added.
- `publications_with_topics.csv` and `publications_with_topics.xlsx` list each publication with its assigned topic, label, and probability score.
- In these publication files, `topic_label` comes from BERTopic’s automatic labels (not the LLM).
- `docs_with_topics_and_macro_topics.csv` is the full document export with topic and macro-topic assignments.
- `hierarchical_topics_raw.csv` contains the hierarchy tree used for the topic dendrogram.
- `hierarchical_topics_labeled.csv` adds LLM labels for each hierarchy node (`LLM_Label`, `LLM_Labels`).
- `hierarchical_topics_tree_llm.html` is a collapsible tree visualization with LLM labels; hover shows additional labels.
- `hierarchical_topics_tree_raw.html` is the same visualization using raw/BERTopic labels (no LLM).
- `topics_hierarchy.html` and `hierarchical_documents.html` are interactive visualizations (when generated).

In the visualization output folder (e.g., `outputs_viz/`):
- `top_topics_<CLUSTER>.png` is a table view of the top topics.
- `treemap_topics_<CLUSTER>.png` shows topic size as a treemap.
- `topic_trends_<CLUSTER>.png` shows topic frequency over time.

## Why This Is Useful
- Faster understanding of large literature collections
- Clear labels for non-technical audiences
- Trend analysis for reporting and policy decisions
- Publication-level transparency for auditing topic assignments

## Limitations
- Topic modeling is unsupervised, so some topics can be mixed or ambiguous.
- Labels are generated by an LLM and can occasionally be off or too generic.
- Labeling only uses the documents that fit into the model context window, not necessarily every document.
- Topic assignments are probabilistic and may be less reliable for short or low-quality abstracts.
- Results depend on the input data and preprocessing choices, so different datasets can yield different topics.

## What Topic -1 Means
In BERTopic, `Topic = -1` represents outliers or noise. These documents did not fit well into any cluster, so they are not assigned to a specific topic.
