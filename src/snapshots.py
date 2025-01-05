import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import pandas as pd
import numpy as np
import networkx as nx
from dask.distributed import LocalCluster
from collections import Counter
import os

df = pd.read_csv("data/plays/Macbeth.csv")

# Count how many times a character speaks in the play and the sentiment across each scene
chars = Counter(df["character"])
scene_sentiments = df.groupby(["act", "scene"])["sentiment"].mean()
sentiment_avg = round(df["sentiment"].mean(), 2)
for key, val in sorted(chars.items(), key=lambda item: item[1], reverse=True):
    print(key, val)
print(scene_sentiments)