# PhySH-Tank : Physics Subject Headings Tag Recommendation System
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://physh-tag-reco.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker_Compose-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Adapter-yellow)](https://huggingface.co/swetlanas/physbert-tag-recommender)

[**LIVE DEMO HERE (Streamlit: physh-tag-reco.streamlit.app/)**](https://physh-tag-reco.streamlit.app/)  


A machine learning app that recommends APS Physical Review B PhySH tags ([Physics Subject Headings](https://github.com/physh-org/PhySH)) for articles based on their titles and abstracts using [PhysBERT](https://huggingface.co/thellert/physbert_uncased) fine-tuned with LoRA (Low Rank Adaptation) across 2000+ labels. Currently optimized for APS Physical Review B articles covering condensed matter and related fields. Check out the [PhySH classification system and its history](https://www.isko.org/cyclo/physh) for background.


## Motivation
When submitting papers to APS journals, authors must manually filter through the 3,000+ concept PhySH taxonomy by discipline and navigate multi-level facet trees (Research Areas, Physical Systems, Properties, Techniques) to select accurate subject headings. Authors are expected to select 3-8 good tags which might be inconsistent across authors and various submissions. This project replaces manual hierarchy browsing by predicting top relevant tags simultaneously across all PRB tags directly from a paper's title and abstract. For comparison, here is how authors currently [browse and select PhySH tags](https://physh.org/tagger).


## Dataset
Model trained on 40,000+ Physical Review B abstracts (volumes 93–113) starting January 2016, when APS switched to using PhySH from PACS, through June 2026. Data collected using [APS Harvest API](https://harvest.aps.org/docs/harvest-api). The dataset is not redistributed in this repository but can be reproduced with the [harvestAPI_pull.py](https://github.com/swetlanas/PhySH-Tank/blob/main/src/data/harvestAPI_pull.py) script  which handles pagination, retries, and checkpointing.. See [`DATACARD.md`](https://github.com/swetlanas/PhySH-Tank/blob/main/docs/DATACARD.md) for full dataset documentation and [`PhySH Taxonomy.md`](https://github.com/swetlanas/PhySH-Tank/blob/main/docs/PhySH%20Taxonomy.md) for a detailed breakdown of the PhySH RDF structure and how it's used in this project.

## Project Structure

```
PhySH-Tank/
├── data                  # Sample json files
│   ├── models            # Saved models
├── docs                  # Documentation
├── literature            # Relevant papers
├── logs                  # Saved logs
├── notebooks             # EDA and experiments
├── src
│   ├── __init__.py
│   ├── data              # Data collection and preprocessing
│   │   ├── deprecated    # Old scraping scripts
│   └── utils             # Helper functions
│   └── models            # ML models
├── app_backend           # FastAPI setup
├── streamlit             # Streamlit cloud app
└── docker-compose.yml    # Container orchestration
```


## Setup

### uv (Local python environment)
```bash
git clone https://github.com/swetlanas/PhySH-Tank  
cd PhySH-Tank  
uv sync
```

### Docker
Run via Docker Compose (Recommended)

```bash
git clone https://github.com/swetlanas/PhySH-Tank.git
cd PhySH-Tank
docker compose up --build
```
Open http://localhost:8501 in your browser


## Model Metrics 

Here the model performance is compared.  
Theoretical baseline is < 1 for precision@5 because many papers have fewer than 5 tags and for recall@5 because many papers have more than 5 tags. Check [02_exploratory_data_analysis](https://github.com/swetlanas/PhySH-Tank/blob/main/notebooks/02_exploratory_data_analysis.ipynb) for tag distribution analysis.  



| Baseline/Model              |Precision@5| Recall@5  |
| --------------------------- | --------- | --------- |
| Theoretical Max             | 0.883     | 0.852     |
| LoRA physBERT               | 0.408     | 0.394     |
| napkinXC + physBERT         | 0.370     | 0.356     |
| physBERT linear head        | 0.363     | —         |
| physBERT + XGBoost          | 0.341     | —         |
| TF-IDF + MultinomialNB      | 0.332     | —         |
| Majority-class baseline     | 0.083     | 0.075     |
| Random                      | ~0.002    | ~0.0017   |


## Roadmap
### v1 — Data pipeline
- [x] BeautifulSoup + Selenium scraper (Original frontend scraping - deprecated)
- [x] `harvestAPI_pull.py` — APS Harvest API, monthly pagination, retry logic, checkpointing
- [x] PhySH RDF graph built with NetworkX (SKOS/DCTERMS/PHYSH namespaces)
- [x] `preprocessing.py` — MathML cleaning (BeautifulSoup/lxml), UUID → tag name mapping
- [x] Baseline model: TF-IDF + MultinomialNB (`MultiOutputClassifier`, `MultiLabelBinarizer`)
- [x] Tag pruning evaluated using recall, F1, precision@k metrics

### v2 — Extreme multi-label classification (XMLC)
The goal is a head-to-head comparison of two distinct ways to solve the
XMLC problem, all evaluated on the same train/test split with the same recsys
metrics (Precision@5, Hit@5, recall@5, NDCG).
 
#### 2a. napkinXC (PLT) on frozen PhysBERT embeddings - Tree Based 
- [x] Baseline napkinXC on TF-IDF
- [x] napkinXC with PhysBERT embeddings

#### 2b. LoRA-fine-tuned PhysBERT classifier (end-to-end) - Transformer Based
- [x] Linear classifier head on top of PhysBERT
- [x] LoRA finetuning for `AutoModelForSequenceClassification` on PhysBERT, `num_labels=3000+`, `problem_type="multi_label_classification"`


### v3 — RecSys evaluation metrics
- [x] NDCG, Recall@k, Hit@K computed incrementally

### v4 — Deployment (demo)
- [x] Streamlit app: title + abstract → Choice of 5 ranked tags 
- [x] Deploy to Streamlit Community Cloud / HuggingFace Spaces

### v5 — Engineering polish 
- [x] FastAPI + Docker



## Future Work
- Seed PLT tree structure with existing PhySH NetworkX graph
- Extend to journal recommendation (PRA, PRB, PRC, PRD, PRE, PRL)
  which requires training data from additional APS journals.
- Two-tower model 
  - [ ] Abstract and text tower (TBD pretrained model) + tag tower (PhysBERT)
  - [ ] Contrastive loss training
  - [ ] FAISS nearest-neighbor retrieval over tag embeddings
  - [ ] Compare frozen-PhysBERT-tower vs. fine-tuned-tower as an ablation
