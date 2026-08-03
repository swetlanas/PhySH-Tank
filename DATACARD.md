# Dataset Card: APS Physics Articles with PhySH Headings (2016–2026)

## 1. Dataset Overview
* **Description:** A dataset of American Physical Society (APS) article metadata, including abstracts, titles, DOIs, and multi-label PhySH (Physics Subject Headings) tags.
* **Timeframe:** January 2016 – June 2026 (Begins when APS migrated from PACS to PhySH).
* **Size:** 50,838 entries (~55,496 raw retrieved).
* **Data Source:** Collected via the [APS Harvest API](https://harvest.aps.org/docs/harvest-api).
* **Primary Task:** Extreme Multi-Label Text Classification (XMLC), Natural Language Processing (NLP), Tag Recommendation Systems.

---
## 2. Dataset Schema
| Field Name | Data Type | Description |
| :--- | :--- | :--- |
| `doi` | `string` | Digital Object Identifier. Access via `doi.org/{doi_value}`. |
| `title` | `string` | Full article title. |
| `abstract` | `string` | Article abstract text. |
| `physh` | `list[string]` | List of PhySH Concept IDs. Map to human-readable names using `src/data/preprocessing.py`. |
| `date` | `datetime64[us]` | Publication timestamp. |
| `articleType` | `string` | APS article category (filtered to 5 core research types). |

---

## 3. Data Collection & Preprocessing

### Key Dataset Statistics
| Field                                            | Value        |
| ------------------------------------------------ | ------------ |
| Total articles retrieved                         | 55,496        |
| Number of duplicates (matching across all metadata fields) dropped        | 4,658         |
| Number of unique PhySH tags in the dataset | 2421         |
| Average number of tags per article               | 5.77 tags/article         |
| Entries with missing PhySH tags (all from 2016)  | 440 (~0.86%) |
|                                                  |              |

APS migrated from PACS to PhySH in 2016. The missing PhySH tags are from this transition period.
More information can be found in the EDA notebook (src/data/02_exploratory_data_analysis.ipynb).
Sample data can be found at data/prb_articles_labeled_Jan2016-Jun2026_duplicate-free_sample.json

### Filtering and preprocessing

**Article Type Filtering:** APS defines 14 article types (`article`, `letter`, `focus`, `viewpoint`, `announcement`, `tutorial`, `erratum`, `feature`, `comment`, `perspective`, `reply`, `review`, `editorial`, `synopsis`). 
  * **Retained (5 Types):** `article`, `letter`, `tutorial`, `perspective`, and `review`.
  * **Excluded (9 Types):** Non-research categories without abstracts or PhySH tags. `erratum` entries were explicitly dropped as they contain duplicate text/tags from original publications.


## 4. Intended Use & Limitations

### Intended Use Cases
* Training supervised multi-label text classifiers on physics domain literature (APS Phy Rev B).
* Evaluating hierarchical tag recommendation engines using the PhySH ontology graph.

### Out-of-Scope Use Cases
* Generative text synthesis of full papers (only abstracts and titles are included).
* Historical analysis prior to 2016 (after introduction of PhySH).

### Known Limitations & Biases
* **Temporal Shift:** PhySH subject headings evolve over time; newer topics introduced after 2016 may have sparse historical representation.
* **Long-Tail Label Distribution:** A small subset of high-frequency tags dominate the dataset, typical of XMLC datasets.