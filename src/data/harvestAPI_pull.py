import requests
import json
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",handlers=[
        logging.FileHandler("harvest.log"),
        logging.StreamHandler()
    ])
logger = logging.getLogger(__name__)

#Define the date range for which API pulls article information
START_DATE = "2016-01-01"
END_DATE = "2026-06-30"


URL="https://harvest.aps.org/v2/journals/articles"

ROOT = Path.cwd()
while not (ROOT / ".git").exists():
    ROOT = ROOT.parent

DATA_PATH = ROOT / "data" / "prb_articles_labeled_Jan2016-Jun2026.json"

HEADERS = {"Accept": "application/vnd.tesseract.article+json"}



def parse_article(data):
    """
    Parses the APS Harvest API json response and extracts the DOI, title, abstract, phySH tags, date, articleType and journal info.
    Only stores articles that don't have the type "synopsis" as those don't have abstracts and tags.

    Args:
    data (json): response from a given URL (and given page)

    Returns:
    tags_list : list of dictionary with relevant keys and values

    """
    valid_article_types = {"article", "letter", "tutorial", "perspective", "review"}

    tags_list = [
        {"doi": article["identifiers"]["doi"], 
        "title" : article['title']['value'], 
        "abstract" : article.get("abstract",{}).get("value",None), 
        "physh" : [concept["id"] for concept in article.get("classificationSchemes",{}).get("physh",{}).get("concepts",{})],
        "date"  : article.get("date",None),
        "articleType" : article.get("articleType",None)
        #"journal" : article["journal"].get("id",None)
        } 
        for article in data["data"]
        if article.get("articleType",None) in valid_article_types and article["journal"].get("id",None)=="PRB"
    ]

    return tags_list


if __name__=="__main__":
    
    start_time = time.time()

    month_starts = pd.date_range(start=START_DATE, end=END_DATE, freq="MS")
    month_ends = pd.date_range(start=START_DATE, end=END_DATE, freq="ME")

    tags_list = [] #Initialize

    pbar = tqdm(zip(month_starts, month_ends), total=len(month_starts))
    
    for month_init, month_final in pbar:

        pbar.set_description(f"Fetching {month_final.strftime('%B %Y')}")
        
        month_start_time = time.time()
        initial_count = len(tags_list)

        params = {
        "from": month_init.strftime("%Y-%m-%d"),
        "until": month_final.strftime("%Y-%m-%d"), 
        "journals": "PRB",
        "date": "published",
        "per_page": 100 #Maximum
        }   

        try:
            logger.info(f"Starting fetch for {params['from']} to {params['until']}")
            response = requests.get(URL, headers=HEADERS, params = params)
            data = response.json()

            tags_list.extend(parse_article(data))

        except requests.exceptions.HTTPError as e:
            logger.error(f"Request failed for {URL}: {e}")


        while True:

            next_url = response.links.get("next",{}).get("url")

            if next_url is None:
                break

            try:    
                time.sleep(0.5)    
                response = requests.get(next_url, headers=HEADERS)
                response.raise_for_status()
                data = response.json()
                tags_list.extend(parse_article(data))
            
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP error: {e}")
                break

            
        month_count = len(tags_list) - initial_count
        month_elapsed = time.time() - month_start_time
        logger.info(f"Fetched {month_count} articles for {month_final.strftime('%B %Y')} in {month_elapsed:.2f} seconds")

        with open(DATA_PATH,'w') as f:
            json.dump(tags_list,f)
    
    end_time = time.time()
    elapsed_time=end_time - start_time
    
    logger.info(f"Total runtime of the program is {elapsed_time:.2f} seconds")
