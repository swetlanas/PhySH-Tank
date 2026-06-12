#src/data/scraper.py

from bs4 import BeautifulSoup
from selenium import webdriver
import pandas as pd
import re
import json
import numpy as np
import logging
logger = logging.getLogger(__name__)



BASE_URL = "https://journals.aps.org"
START_VOLUME = 93
OUTPUT_PATH = "data/raw/prb_tags.csv"


def generate_all_article_url(total_issues = 24,total_volumes = 18):
    """
    Loops through each volume and issue to collect all the article URLS in a given Volume and Issue.
    Journal -> Volume -> Issue -> Article
    Considering 24 issues and 18 volumes in each issue by default.

    Args:
        total_issues (int) : Number of journal issues wanted
        total_volume (int) : Number of journal volumes wanted

    Returns:
        list of all article URLs


    """

    #Create a list of urls for each issue in a given volume of APS Physical Review B starting in January 2016 (VOLUME 93) when they started using PhySH tags
    journal_issue_urls=[BASE_URL+"/prb/issues/"+str(volume_idx+START_VOLUME)+"/"+str(issue_idx+1) for volume_idx in range(total_volumes) for issue_idx in range(total_issues)]

    

    journal_dictionary = {}
    
    driver = webdriver.Chrome()
    for idx in range(len(journal_issue_urls)):
        driver.get(journal_issue_urls[idx]) #idx loops over each issue
        html_source=driver.page_source

        soup = BeautifulSoup(html_source, "html.parser")
        journal_dictionary[f'raw_article_tags{idx}']=soup.find_all('a',class_="default-link-no-flex heading-base-bold") 
        #Get all article URLs for a given journal
        journal_dictionary[f'article_urls{idx}']=[BASE_URL+tags.get('href') for tags in list(journal_dictionary[f'raw_article_tags{idx}'])]

    driver.quit()
    journal_article_urls_all = np.concatenate([journal_dictionary[f'article_urls{idx}'] for idx in range(total_issues*total_volumes)])
    
    return journal_article_urls_all



def scrape_abstract_tags(urls):
    """
    Scrape abstract and PhySH tags for a given article

    Args:
        urls : List of URLs to be scraped
        
    Returns:
        abstracts_list : list of abstracts found
        phys_labels : list of PhySH tags

    """
    
    driver = webdriver.Chrome()

    abstracts_list=[]
    phys_labels_list=[]

    for url in urls:
        driver.get(url)
        html_source_articles=driver.page_source
        soup_articles = BeautifulSoup(html_source_articles, "html.parser")
        #Abstract text for each article
        try:
            abstract_text=soup_articles.find('div',{'id':'abstract-section-content'}).find('p')
            if abstract_text is not None:
                abstracts_list.append(abstract_text.get_text())
                #Get the Physics Subject Headings for a given article
                phys_labels_list.append([tags.get_text() for tags in soup_articles.find_all("a", class_="physh-concept")])
                abstract_text.decompose()

        except:
            logger.warning(f"No PhySH tags found for {url}")


    logger.info(f"Scraped {len(abstracts_list)} abstracts successfully")
    logger.info(f"The length of phys_labels list is {len(phys_labels_list)}")

    
    driver.quit() 

    return abstracts_list, phys_labels_list

def save_to_file(abstracts_list, phys_labels_list,output_path=OUTPUT_PATH):
    """
    Save scraped abstracts and PhySH tags to CSV, filtering out articles with no tags.

    Args:
        abstracts_list: List of abstract strings
        phys_labels_list: List of PhySH tag lists
        output_path: Path to save the CSV file

    Returns:
        None — saves file to disk
    """
    try:
        data = {'Abstracts': abstracts_list, 'PhysHeadings': phys_labels_list}
        df=pd.DataFrame(data)

        #Remove articles without physics subject headings
        df=df[df['PhysHeadings'].map(len) > 0]
        df.reset_index(drop=True, inplace=True)

        #Save to file
        df.to_csv(output_path)

    except:
        print('No articles were scraped. Dataframe is empty.')

    return None


if __name__ == "__main__":
    urls_all = generate_all_article_url(total_issues =1,total_volumes=2)
    abstracts_list, phys_labels_list = scrape_abstract_tags(urls_all)
    save_to_file(abstracts_list, phys_labels_list)

