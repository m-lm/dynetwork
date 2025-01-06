import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
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

# Get list of normalized character names to add to NER ruleset
characters = [name for name in df["character"].unique().tolist() if name != "All"]
original_character_map = {name.lower(): name for name in characters}
characters_normalized = list(map(str.lower, characters))
patterns = [{"label": "PERSON", "pattern": name} for name in characters]
ruler.add_patterns(patterns)

# Extract nonempty entity sets for sentence texts
df["entities"] = df["text"].apply(lambda x: [ent.text.translate(str.maketrans("", "", string.punctuation)) for ent in nlp(x).ents if ent.label_ == "PERSON"]).to_list()
df_filtered = df[df["entities"].apply(len) > 0]

# Per scene, get scene character cast and first check for explicit mentions.
# Then, move on to implicit relations from scene associations, making sure not to double up
# if a speaker directly addresses a fellow scene participant
scenes = df.groupby(["act", "scene"])
explicit_relation_counts = Counter()
implicit_relation_counts = Counter()
for (act, scene), lines in scenes:
    scene_chars = [name for name in lines["character"].unique() if name != "All"]
    associations = [] # used to keep track of explicit relations in a scene when checking implicit relations

    # Organize the relationships pairwise according to a commutative/undirected (1st party, 2nd party) order
    # This is for explicit mentions by a speaker
    for _, row in lines.iterrows():
        mentioned = row["entities"]
        speaker = row["character"]

        # Skip if the speaker is "All"; only want individuals
        if speaker == "All":
            continue

        for mention in mentioned:
            if speaker != mention and mention != "All":
                ordered_pair = (speaker, mention)
                unordered_pair = tuple(sorted((speaker, mention)))
                associations.append(unordered_pair)
                explicit_relation_counts[unordered_pair] += 1

    # Factor in implicit relationships via scene associations. Characters within the same scene likely are related.
    for combo in itertools.combinations(scene_chars, 2):
        ordered_pair = combo
        unordered_pair = tuple(sorted(combo))
        # Prioritize explicit mentions over implicit scene associations
        if unordered_pair not in associations:
            associations.append(unordered_pair)
            implicit_relation_counts[unordered_pair] += 1

total_counts = explicit_relation_counts + implicit_relation_counts
for key, _ in sorted(total_counts.items(), key=lambda item: item[1], reverse=False):
    print("KEY: ", key)
    print("TOTAL: ", total_counts[key])
    print("EXPLICIT: ", explicit_relation_counts[key])
    print("IMPLICIT: ", implicit_relation_counts[key])
    print("\n"*2)

print(scenes["character"].unique()) # print each scene and their explicit entities mentionec

# Visualize character relations as signed weighed graph
G = nx.DiGraph()

for relation, weight in total_counts.items():
    G.add_edge(relation[0], relation[1], weight=weight)

viz = Network(
    notebook=True, 
    cdn_resources="remote",
    neighborhood_highlight=True,
    select_menu=True,
    filter_menu=True,
    )
viz.toggle_physics(True)
for node in G.nodes():
    viz.add_node(node, label=node, size=G.degree(node))
viz.from_nx(G)
viz.show("viz/emonet.html")