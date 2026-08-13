# PhySH-Tank : Physics Subject Headings Tag Recommendation System
ML model that recommends PhySH tags ([Physics Subject Headings](https://github.com/physh-org/PhySH)) based on journal abstract. Currently optimized for APS Physical Review B articles covering condensed matter and related fields. Learn more about the PhySH classification system and its history [here](https://www.isko.org/cyclo/physh).


## Motivation
When submitting papers to APS journals, authors must manually select 
relevant tags from 260+ PhySH subject headings which is a time-consuming 
and ineffecient process. Authors are expected to select 3-8 good tags which might be inconsistent across authors and various submissions. This project automates the tag selection process by recommending relevant tags based on the article's abstract.


## Dataset
Model trained on 40,000+ Physical Review B abstracts (volumes 93–110) starting January 2016, when APS switched to using PhySH from PACS, through December 2024. Abstracts collected prior to Cloudflare protection which prevents automated scraping. The dataset is not redistributed in this repository.

## Project Structure

```
PhySH-Tank/
├── src/
│   ├── data/          # data collection and preprocessing
│   └── models/        # ML models
├── notebooks/         # EDA and experiments  
├── data/              # Sample json files
└── app/               # Streamlit demo 
```


## Setup

### Recommended (uv)
```
git clone https://github.com/swetlanas/PhySH-Tank  
cd PhySH-Tank  
uv sync
```

### pip
```
pip install -r requirements.txt
```

### conda
```
conda create -n physh-env python=3.12.3  
conda activate physh-env  
pip install -r requirements.txt
```

## Roadmap
### v1 — Data pipeline
- [x] BeautifulSoup + Selenium scraper (Original frontend scraping)
- [x] `harvest_scraper.py` — APS Harvest API, monthly pagination, retry logic, checkpointing
- [x] PhySH RDF graph built with NetworkX (SKOS/DCTERMS/PHYSH namespaces)
- [x] `preprocessing.py` — MathML cleaning (BeautifulSoup/lxml), UUID → tag name mapping
- [x] Baseline model: TF-IDF + MultinomialNB (`MultiOutputClassifier`, `MultiLabelBinarizer`)
- [x] Tag pruning evaluated using recall, F1, precision@k metrics

### v2 — Extreme multi-label classification (XMLC): Three comparable approaches
The goal is a **head-to-head comparison** of three ways to solve the
XMLC problem, all evaluated on the same train/test split with the same
metrics (precision@5, Hit@5, MRR, NDCG).
 
#### 2a. napkinXC (PLT) on frozen PhysBERT embeddings
- [x] Baseline napkinXC on TF-IDF
- [x] napkinXC with PhysBERT embeddings
- [ ] Seed PLT tree structure with existing PhySH NetworkX graph

#### 2b. LoRA-fine-tuned PhysBERT classifier (end-to-end)
- [x] Linear classifier head on top of PhysBERT
- [ ] `AutoModelForSequenceClassification` on PhysBERT, `num_labels=3000+`, `problem_type="multi_label_classification"`


#### 2c. Two-tower model 
- [ ] Abstract and text tower (TBD pretrained model) + tag tower (PhysBERT)
- [ ] Contrastive loss training
- [ ] FAISS nearest-neighbor retrieval over tag embeddings
- [ ] Compare frozen-PhysBERT-tower vs. fine-tuned-tower as an ablation

### v2.5 — Journal recommendation head (maybe)
- [ ] Multi-task extension, scoped to PRB (single-journal training data)

### v3 — RecSys evaluation metrics
- [ ] NDCG, MRR, Hit@K computed incrementally

### v4 — Deployment (demo)
- [ ] Streamlit app: title + abstract → tags + journal
- [ ] Deploy to Streamlit Community Cloud / HuggingFace Spaces

### v5 — Engineering polish (maybe)
- [ ] FastAPI + Docker



## Future Work
- Extend to journal recommendation (PRA, PRB, PRC, PRD, PRE, PRL)
  which requires training data from additional APS journals.
