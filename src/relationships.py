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
plays = sorted(os.listdir("data/plays"))

# Read each play CSV and extract relationships
for play in plays:
    if play[0].isalpha():
        filename = os.path.join("data/plays", play)
    else:
        continue

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
        # From here, lower() is used for name normalization. At the end the character names are mapped back to proper case.
        # This is for explicit mentions by a speaker
        for _, row in lines.iterrows():
            mentioned = row["entities"]
            speaker = row["character"].lower()

            # Skip if the speaker is "All"; only want individuals
            if speaker == "all":
                continue

            # Iterate through mentioned entities from sentence text. Note that these entities are normalized to lowercase.
            for mention in list(map(str.lower, mentioned)):
                if speaker != mention and mention != "all":
                    ordered_pair = (speaker, mention)
                    unordered_pair = tuple(sorted((speaker, mention)))
                    associations.append(unordered_pair)
                    explicit_relation_counts[unordered_pair] += 1

        # Factor in implicit relationships via scene associations. Character within the same scene likely are related.
        for combo in itertools.combinations(list(map(str.lower, scene_chars)), 2):
            ordered_pair = combo
            unordered_pair = tuple(sorted(combo))
            # Prioritize explicit mentions over implicit scene associations
            if unordered_pair not in associations:
                associations.append(unordered_pair)
                implicit_relation_counts[unordered_pair] += 1

    # Convert lowercase normalized names to original proper case and check if gathered names are in the list of play characters who participate
    # Note this primarily only works for plays as we have a convenient list of characters in the play and the worldbuilding does not cross this boundary.
    # However, longer more complex works like epic fantasy or scifi should include those nonpresent 3rd parties mentioned even if they do not actively participate. This is for worldbulding reasons.
    explicit_relation_counts = Counter({(original_character_map[a], original_character_map[b]): count for (a, b), count in explicit_relation_counts.items() if a in original_character_map and b in original_character_map})
    implicit_relation_counts = Counter({(original_character_map[a], original_character_map[b]): count for (a, b), count in implicit_relation_counts.items() if a in original_character_map and b in original_character_map})
    total_counts = explicit_relation_counts + implicit_relation_counts
    for key, _ in sorted(total_counts.items(), key=lambda item: item[1], reverse=False):
        print("KEY: ", key)
        print("TOTAL: ", total_counts[key])
        print("EXPLICIT: ", explicit_relation_counts[key])
        print("IMPLICIT: ", implicit_relation_counts[key])
        print("\n"*2)

    print(scenes["character"].unique()) # print each scene and their explicit entities mentioned

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
    for node in G.nodes():
        viz.add_node(node, label=node, size=G.degree(node))
    viz.from_nx(G)

    play_title = play[:play.find(".")]
    viz.show(f"viz/{play_title}.html")
    print(f"{play_title} done processing...")
    break