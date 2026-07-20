#src/data/preprocessing.py

#Maps concept IDs to PhySH tag names, cleans up abstract and title to make it ready for model training


from pathlib import Path
import pandas as pd
import requests
import re
import networkx as nx
import numpy as np
from rdflib import Graph, Namespace
import gzip,io
import logging
from functools import lru_cache

ROOT = Path.cwd()
while not (ROOT / ".git").exists():
    ROOT = ROOT.parent
DATA = ROOT / "data" / "prb_headings_full.csv"

URL = "https://raw.githubusercontent.com/physh-org/PhySH/master/physh.rdf.gz"
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler()
    ])
logger = logging.getLogger(__name__)






def concept_id_to_physh_name():
    """
    Loads the concept ID to tag name mapping from PhySH github and returns a dictionary with concept ID keys and PhySH tag name values.
    """

    response = requests.get(URL, stream=True)
    with gzip.open(io.BytesIO(response.content), "rt", encoding="utf-8") as f:
            rdf_text = f.read()

    g = Graph()
    g.parse(data=rdf_text, format="xml")


    uuid_to_label = {}
    for subject, predicate, obj in g.triples((None, SKOS.prefLabel, None)):
            if obj.language == "en":
                uuid = str(subject).split("/")[-1]
                uuid_to_label[uuid] = str(obj)
    
    return uuid_to_label

def concept_hierarchy_graph():
    """
    Loads PhySH rdf from github and returns a NetworkX graph with concept ID hierarchy.
    """

    response = requests.get(URL, stream=True)
    with gzip.open(io.BytesIO(response.content), "rt", encoding="utf-8") as f:
            rdf_text = f.read()

    g = Graph()
    rdf_data = g.parse(data=rdf_text, format="xml")

    nx_graph = nx.DiGraph()

    # Build directional edges: Parent -> Child
    for child,predicate, parent in g.triples((None, SKOS.broader, None)):
        child_id = str(child)
        parent_id = str(parent)
        nx_graph.add_edge(parent_id, child_id)

    # Build directional edges: Child -> Parent
    for child,predicate, parent in g.triples((None, SKOS.narrower, None)):
        child_id = str(child)
        parent_id = str(parent)
        nx_graph.add_edge(child_id, parent_id) #add_edge() automatically takes care of duplicates if a parent, child pair was already defined

    # Add node labels 
    for child,predicate, parent in g.triples((None, SKOS.prefLabel, None)):
        if str(child) in nx_graph:
            nx_graph.nodes[str(child)]['label'] = str(parent)
    
    return nx_graph

def clean_abstract(df_raw):
    """
    Cleans up the abstract text for training.

    Args:
    df_raw (dataframe) : input dataframe containing abstracts and phySH tags

    Returns:
    df_cleaned (dataframe) : Cleaned-up dataframe

    """

    #Removes the first column which encodes index information
    df_raw.drop(columns=df_raw.columns[0], inplace=True)

    #If phySH tags are empty, replace it with empty list
    df_raw['PhysHeadings'] = df_raw["PhysHeadings"].fillna("[]").apply(lambda x: eval(x))

    #Remove duplicate abstracts
    df_cleaned = df_raw.drop_duplicates(subset=['Abstracts'],keep="first")

    return df_cleaned



if __name__=='__main__':
    concept_id_map = concept_id_to_physh_name()
    
    #df = pd.read_csv(DATA,sep=',', engine='python')

