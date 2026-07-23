#src/data/preprocessing.py

#Maps concept IDs to PhySH tag names, cleans up abstract and title to make it ready for model training


from pathlib import Path
import pandas as pd
import requests
import re
import networkx as nx
import numpy as np
from rdflib import Graph, Namespace,SKOS, RDF, URIRef
import gzip,io
import logging
from functools import lru_cache

ROOT = Path.cwd()
while not (ROOT / ".git").exists():
    ROOT = ROOT.parent
DATA = ROOT / "data" / "prb_headings_full.csv"


URL = "https://raw.githubusercontent.com/physh-org/PhySH/master/physh.rdf.gz"
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
PHYSH = Namespace("https://physh.org/rdf/2018/01/01/core#")
DCTERMS = Namespace("http://purl.org/dc/terms/")


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler()
    ])
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_physh_graph():
    """
    Downloads and parses the PhySH rdf graph from Github.

    returns:
    g (rdflib.Graph) : parsed PhySH concept graph

    raises:
    requests.exceptions.HTTPError : if download fails
    """
    try:
        response = requests.get(URL, stream=True)
        response.raise_for_status()
        with gzip.open(io.BytesIO(response.content), "rt", encoding="utf-8") as f:
                rdf_text = f.read()

        g = Graph()
        g.parse(data=rdf_text, format="xml")
        return g
    
    except requests.exceptions.HTTPError as e:
        logger.error(f"Request failed for {URL}: {e}")
        raise RuntimeError(f"Failed to load PhySH graph: {e}") from e
        



def concept_id_to_physh_name(graph=load_physh_graph()):
    """
    Loads the concept ID to tag name mapping from PhySH github and returns a dictionary with concept ID keys and PhySH tag name values.
    """

    uuid_to_label = {}
    for subject, predicate, obj in graph.triples((None, SKOS.prefLabel, None)):
            if obj.language == "en":
                uuid = str(subject).split("/")[-1]
                uuid_to_label[uuid] = str(obj)
    
    return uuid_to_label

def concept_hierarchy_graph(graph=load_physh_graph()):
    """
    Loads PhySH rdf from github and returns a NetworkX graph with concept ID hierarchy using both SKOS and custom PHYSH.
    """

    nx_graph = nx.DiGraph()
    #add_edge() automatically takes care of duplicates if a parent, child pair was already defined
    #add_edge(u_source, v_target) makes u -> v

    #Initialize tree with all concepts, disciplines and facets as nodes.
    for subject in graph.subjects():
        if isinstance(subject,URIRef):
             nx_graph.add_node(str(subject))

    # SKOS directional/hierarchical edges: Parent -> Child
    for child,predicate, parent in graph.triples((None, SKOS.broader, None)):
        child_id = str(child)
        parent_id = str(parent)
        nx_graph.add_edge(parent_id, child_id)

    # SKOS directional/hierarchical edges: Child -> Parent
    for child,predicate, parent in graph.triples((None, SKOS.narrower, None)):
        child_id = str(child)
        parent_id = str(parent)
        nx_graph.add_edge(child_id, parent_id) 

    # PhySH Facet structural edges: Parent (Facet) -> Child (Concept)
    for child,predicate, parent in graph.triples((None, PHYSH.inFacet, None)):
        child_id = str(child)
        parent_id = str(parent)
        nx_graph.add_edge(parent_id, child_id)

    # PhySH Discipline structural edges: Parent (Discipline) -> Child (Concept)
    for child,predicate, parent in graph.triples((None, PHYSH.inDiscipline, None)):
        child_id = str(child)
        parent_id = str(parent)
        nx_graph.add_edge(parent_id, child_id)

    # Container (Parent) -> Concept (Child)
    for child,predicate, parent in graph.triples((None, PHYSH.hasConcept, None)):
        child_id = str(child)
        parent_id = str(parent)
        nx_graph.add_edge(child_id,parent_id)

    for child,predicate, parent in graph.triples((None, PHYSH.contains, None)):
        child_id = str(child)
        parent_id = str(parent)
        nx_graph.add_edge(child_id,parent_id)

    # Add node labels from SKOS
    for child,predicate, parent in graph.triples((None, SKOS.prefLabel, None)):
        if str(child) in nx_graph:
            nx_graph.nodes[str(child)]['label'] = str(parent)

    # Add node labels from dcterms for Disciplines
    for child,predicate, parent in graph.triples((None, DCTERMS.title, None)):
            if str(child) in nx_graph:
                nx_graph.nodes[str(child)]['label'] = str(parent)

    # Add deprecation status
    for child,predicate,parent in graph.triples((None, PHYSH.deprecated, None)):
        if str(child) in nx_graph:
            nx_graph.nodes[str(child)]['deprecated'] = (str(parent).lower() == 'true')
    
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

