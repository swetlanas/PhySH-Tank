#src/data/preprocessing.py

#Maps concept IDs to PhySH tag names, loads graph structure of PhySH tags, cleans up abstract and title to make it ready for model training


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
from bs4 import BeautifulSoup


ROOT = Path.cwd()
while not (ROOT / ".git").exists():
    ROOT = ROOT.parent
DATA = ROOT / "data" / "prb_articles_labeled_Jan2016-Jun2026_duplicate-free.json"


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
        

def concept_hierarchy_graph(graph=None):
    """
    Loads PhySH rdf from github and returns a NetworkX graph with concept ID hierarchy using both SKOS and custom PHYSH.

    Args: 
        graph (rdf) : rdf graph file to be parsed

    Returns: 
        nx_graph (nx.DiGraph()) : parsed NetworkX graph consisting of node labels and edges
    """

    if graph is None:
        graph = load_physh_graph()

    nx_graph = nx.DiGraph()
    #add_edge() automatically takes care of duplicates if a parent, child pair was already defined
    #add_edge(u_source, v_target) makes u -> v

    #Initialize tree with all concepts, disciplines and facets as nodes.
    for subject in graph.subjects():
        if isinstance(subject,URIRef):
             nx_graph.add_node(str(subject))

        
    for child,predicate, parent in graph:
        child_id = str(child)
        parent_id = str(parent)
        # SKOS directional/hierarchical edges: Parent -> Child
        if predicate == SKOS.broader:
            nx_graph.add_edge(parent_id, child_id,rel_type="SKOS concept parent")
        
        # SKOS directional/hierarchical edges: Child -> Parent
        elif predicate == SKOS.narrower:
            nx_graph.add_edge(child_id, parent_id, rel_type="SKOS concept parent")

        # PhySH Facet structural edges: Parent (Facet) -> Child (Concept)
        elif predicate == PHYSH.inFacet:
            nx_graph.add_edge(parent_id, child_id, rel_type="PHYSH Facet parent")

        # PhySH Discipline structural edges: Parent (Discipline) -> Child (Concept)
        elif predicate == PHYSH.inDiscipline:
            nx_graph.add_edge(parent_id, child_id, rel_type="PHYSH Discipline parent")

        # Concept (Parent) -> Container (Child)
        elif predicate == PHYSH.hasConcept:
            nx_graph.add_edge(parent_id,child_id, rel_type="PHYSH Discipline parent")
    
        elif predicate == PHYSH.contains:
            nx_graph.add_edge(parent_id,child_id, rel_type="PHYSH Facet parent")

        # ---Node labels---
        # Preferred Label from SKOS
        elif predicate==SKOS.prefLabel and child_id in nx_graph:
            nx_graph.nodes[child_id]['label'] = parent_id
        # Title from DCTERMS (Fallback for Disciplines/Facets)
        elif predicate==DCTERMS.title and child_id in nx_graph:
            #nx_graph.nodes[str(child)]['label'] = str(parent)
            nx_graph.nodes[child_id].setdefault('label', parent_id)
        # Add deprecation status
        elif predicate==PHYSH.deprecated and child_id in nx_graph:
            nx_graph.nodes[child_id]['deprecated'] = (parent_id.lower() == 'true')
        
    return nx_graph

def clean_abstract(raw_abstract_text):
    """
    Cleans up the abstract text containing Math XML / HTML for training.

    Args:
    raw_abstract_text (str) : input abstract text containing HTML and Math XML

    Returns:
    abstract_cleaned (str) : Cleaned-up abstract text

    """

    soup = BeautifulSoup(raw_abstract_text, "lxml")
    abstract_cleaned = soup.get_text(separator="").strip()

    return abstract_cleaned

def prep_data(df):

    #Makes NetworkX graph for physh tags
    graph_hierarchy = concept_hierarchy_graph()
    label_naming = nx.get_node_attributes(graph_hierarchy,'label')

    #Applying it to the concept IDs in the dataset and mapping it to PhySH tag names.
    df["physh_names"]=df["physh"].apply(
        lambda uuid_list: [
            label_naming.get(PHYSH_URL_PREFIX+uuid,None) for uuid in uuid_list
        ]
    )

    #Parsing XML/HTML and cleaning up the abstract text
    df['title']=df['title'].apply( lambda x : clean_abstract(x))

    #Parsing XML/HTML and cleaning up the abstract text
    df['abstract']=df['abstract'].apply( lambda x : clean_abstract(x))

    #If phySH tags are empty, drop that entry
    df_cleaned = df[~df['physh'].apply( lambda x : not x)]

    return df_cleaned



if __name__=='__main__':
    df = prep_data(pd.read_json(DATA))
    


