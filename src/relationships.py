import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import pandas as pd
import numpy as np
import networkx as nx
from dask.distributed import LocalCluster
from collections import Counter
import os
import string
import itertools

# Setup
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("spacytextblob")
ruler = nlp.add_pipe("entity_ruler", before="ner")
df = pd.read_csv("data/plays/Macbeth.csv")

# Get list of character names to add to NER ruleset
characters = df["character"].unique().tolist()
patterns = [{"label": "PERSON", "pattern": name} for name in characters]
ruler.add_patterns(patterns)

# Extract nonempty entity sets for sentence texts
df["entities"] = df["text"].apply(lambda x: [ent.text.translate(str.maketrans("", "", string.punctuation)) for ent in nlp(x).ents if ent.label_ == "PERSON"]).to_list()
df_filtered = df[df["entities"].apply(len) > 0]

# Organize the relationships pairwise according to a commutative/undirected (1st party, 2nd party) order
# This is for explicit mentions by a speaker
explicit_relations = []
for _, row in df_filtered.iterrows():
    cooccurrences = row["entities"]
    speaker = row["character"]
    for c in cooccurrences:
        if speaker != c:
            explicit_relations.append((speaker, c))
explicit_relations = [tuple(pair) for pair in list(map(sorted, explicit_relations))]

# Factor in implicit relationships via scene associations. Characters within the same scene likely are related.
scene_chars = df.groupby(["act", "scene"])["character"].unique()
implicit_relations = []
for char in scene_chars:
    print(char)
    for combo in itertools.combinations(c, 2):
        pair = tuple(sorted(combo))
        # Prioritize explicit mentions over implicit scene associations
        if pair not in explicit_relations:
            implicit_relations.append(pair)

total_relations = implicit_relations + explicit_relations
total_counts = Counter(total_relations)
print(total_counts)

df_relations = pd.DataFrame(total_relations, columns=["POV", "Mentioned"])
print(df_relations)