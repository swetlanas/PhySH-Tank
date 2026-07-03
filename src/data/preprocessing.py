#src/data/preprocessing.py


from pathlib import Path
import pandas as pd
import re

ROOT = Path.cwd()
while not (ROOT / ".git").exists():
    ROOT = ROOT.parent
DATA = ROOT / "data" / "prb_headings_full.csv"


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
    df = pd.read_csv(DATA,sep=',', engine='python')

