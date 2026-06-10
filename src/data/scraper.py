#src/data/scraper.py

from bs4 import BeautifulSoup
from selenium import webdriver
import pandas as pd
import regex as re
import json
import numpy as np



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
    journal_issue_urls=["https://journals.aps.org/prb/issues/"+str(volume_idx+93)+"/"+str(issue_idx+1) for volume_idx in range(total_volumes) for issue_idx in range(total_issues)]

    

    journal_dictionary = {}

    for idx in range(len(journal_issue_urls)):
        driver = webdriver.Chrome()
        driver.get(journal_issue_urls[idx]) #idx loops over each issue
        html_source=driver.page_source

        soup = BeautifulSoup(html_source, "html.parser")
        journal_dictionary[f'raw_article_tags{idx}']=soup.find_all('a',class_="default-link-no-flex heading-base-bold") 
        #Get all article URLs for a given journal
        journal_dictionary[f'article_urls{idx}']=[str("https://journals.aps.org")+tags.get('href') for tags in list(journal_dictionary[f'raw_article_tags{idx}'])]

    journal_article_urls_all = np.concatenate([journal_dictionary[f'article_urls{idx}'] for idx in range(total_issues*total_volumes)])
    
    return journal_article_urls_all



def scrape_title_abstract_tags(urls):
    """
    Scrape title, abstract and PhySH tags for a given article

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
            print(f"No phySH tags for '{url}'.") 

    print(f"The length of abstract list is {len(abstracts_list)}")
    print(f"The length of phys_labels list is {len(phys_labels_list)}")
    
    return abstracts_list, phys_labels_list

def save_to_file(abstracts_list, phys_labels_list):
    """
    Removing article with no PhySH tags
    """
    try:
        data = {'Abstracts': abstracts_list, 'PhysHeadings': phys_labels_list}
        df=pd.DataFrame(data)

        #Remove articles without physics subject headings
        df=df[df['PhysHeadings'].map(len) > 0]
        df.reset_index(drop=True, inplace=True)

        #Save to file
        df.to_csv('./data/prb_tags.csv')

    except:
        print('No articles were scraped. Dataframe is empty.')

    return None


if __name__ == "__main__":
    urls_all = generate_all_article_url(total_issues =1,total_volumes=2)
    abstracts_list, phys_labels_list = scrape_title_abstract_tags(urls_all)
    save_to_file(abstracts_list, phys_labels_list)
    pass
