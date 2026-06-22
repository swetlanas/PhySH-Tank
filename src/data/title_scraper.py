import json
import re
import pandas as pd
import requests
from pathlib import Path


ROOT = Path.cwd()
while not (ROOT / ".git").exists():
    ROOT = ROOT.parent

DATA_PATH = ROOT / "data" / "all_article_urls.json"


def extract_doi(url):
    """
    Extracts the DOI from the article URL

    Args:
        url (string) : URL
    
    Returns:
        DOI as a string
    """
    m = re.search(r"10\.1103/[^\s]+", url)
    return m.group(0) if m else None


def scrape_title(url):
    
    """
    Uses SemanticScholar API to request for title and abstract

    Args:
        url (string) : URL
    
    Returns:
        dictionary of title and abstract

    """

    params = {"fields": "title,abstract"}

    r = requests.get(url, params=params)
    if r.status_code != 200:
        print('OK')

    data = r.json()

    return {
        "title": data.get("title"),
        "abstract": data.get("abstract")
    }

if __name__ == "__main__":
    # Read json containing all article URLs into a dataframe.
    df = pd.read_json(DATA_PATH)
    pd.set_option('display.max_colwidth', None)
    df.rename(columns={0 : 'URL'},inplace=True)

    #Make a list of SemanticScholar URLs using the DOI
    doi_list= df['URL'].apply(extract_doi).to_list()
    doi_url_list = [f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}" for doi in doi_list]

    data = [scrape_title(url) for url in doi_url_list[0:15]]

    with open(DATA_PATH.parent / "article_titles_abstracts.json", "w") as f:
        json.dump(data, f, indent=2)



