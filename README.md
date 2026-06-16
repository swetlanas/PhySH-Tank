# PhySH-Tank : Physics Subject Headings Tag Recommendation System
ML model that recommends PhySH tags ([Physics Subject Headings](https://github.com/physh-org/PhySH)) based on journal abstract. Currently optimized for APS Physical Review B articles covering condensed matter and related fields.


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
└── app/               # Streamlit demo 
```


## Setup
    conda env create -f environment.yml
    conda activate physhtank

## Roadmap
- [x] v1 — Web scraper (BeautifulSoup + Selenium)
- [ ] v2 — PhySH REST API client + concept graph
- [ ] v3 — Baseline models (TF-IDF, Word2Vec)
- [ ] v4 — Two-tower retrieval model (PyTorch)
- [ ] v5 — SciBERT fine-tuning (HuggingFace)
- [ ] v6 — Streamlit demo app

## Future Work
- Add article title information to training set.
- Extend to journal recommendation (PRA, PRB, PRC, PRD, PRE, PRL)
  which requires training data from additional APS journals.
