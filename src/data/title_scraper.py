import json
import re
import pandas as pd
import requests

def extract_doi(url):
    m = re.search(r"10\.1103/[^\s]+", url)
    return m.group(0) if m else None


def scrape_title(url):
    
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
    df = pd.read_json('/home/rustycutlery/cppexampl/pys/PhySH-Tank/data/all_article_urls.json')
    pd.set_option('display.max_colwidth', None)
    df.rename(columns={0 : 'URL'},inplace=True)

    #Make a list of SemanticScholar URLs using the DOI
    doi_list= df['URL'].apply(extract_doi).to_list()
    doi_url_list = [f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}" for doi in doi_list]

    data = [scrape_title(url) for url in doi_url_list[0:15]]

    with open("/home/rustycutlery/cppexampl/pys/PhySH-Tank/data/article_titles_abstracts.json", "w") as f:
        json.dump(data, f, indent=2)



