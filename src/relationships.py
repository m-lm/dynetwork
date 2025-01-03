import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import pandas as pd
import numpy as np
import networkx as nx
from dask.distributed import LocalCluster
import string
from collections import Counter

nlp = spacy.load("en_core_web_trf")
nlp.add_pipe("spacytextblob")
csv = pd.read_csv("data/plays/Macbeth.csv", chunksize=1)

pols = []
chars = []
for line in csv:
    doc = nlp(line["text"].to_string(index=False))
    #print(f"{round(doc._.blob.polarity, 2)} {doc}")
    chars.append(line["character"].to_string(index=False))
    pols.append(round(doc._.blob.polarity, 2))

chars = Counter(chars)
avg_pol = round(np.mean(pols), 2)
print(f"Avg sentiment: {avg_pol}")
print(chars)