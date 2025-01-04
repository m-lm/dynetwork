import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import pandas as pd
import numpy as np
import networkx as nx
from dask.distributed import LocalCluster
from collections import Counter
import os

# Setup
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("spacytextblob")
plays = sorted(os.listdir("data/plays"))
def get_sentiment(text):
    # Return the polarity of the sentence text
    doc = nlp(str(text))
    return round(doc._.blob.polarity, 2)

# Read the CSV data, calculate the sentiments of each row's sentence, and add
# the sentiment column to the CSV file

for play in plays:
    if play[0].isalpha():
        filename = os.path.join("data/plays", play)
    else:
        continue
    df = pd.read_csv(filename)
    df["sentiment"] = df["text"].apply(get_sentiment)
    df.to_csv(filename, index=False)

# Count how many times a character speaks in the play
chars = Counter(df["character"])
for key, val in sorted(chars.items(), key=lambda item: item[1], reverse=True):
    print(key, val)

sentiment_avg = round(df["sentiment"].mean(), 2)
print(sentiment_avg)