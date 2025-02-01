import spacy
import pandas as pd
import networkx as nx
from pyvis.network import Network
from collections import Counter
from tqdm import tqdm
import os
import string
import itertools
import json
import time

# Setup
nlp = spacy.load("en_core_web_sm")
ruler = nlp.add_pipe("entity_ruler", before="ner")
plays = sorted(os.listdir("data/plays"))

def extract_relationships():
    # Read each play CSV and extract relationships
    cumulative_elapsed_time = 0
    with tqdm(total=len(plays), desc="Processing plays", dynamic_ncols=True, position=0) as pbar:
        for play in plays:
            start_time = time.time()
            if play[0].isalpha():
                filename = os.path.join("data/plays", play)
            else:
                continue

            df = pd.read_csv(filename)

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
            shadow_dict = {} # Create a shadow dict to keep track of temporal info during the loop about the relation count
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

                # Update temporal info graph snapshots of relations and their weights for each scene
                shadow_exp_rel_counts = Counter({(original_character_map[a], original_character_map[b]): count for (a, b), count in explicit_relation_counts.items() if a in original_character_map and b in original_character_map})
                shadow_imp_rel_counts = Counter({(original_character_map[a], original_character_map[b]): count for (a, b), count in implicit_relation_counts.items() if a in original_character_map and b in original_character_map})
                shadow_total_counts = shadow_exp_rel_counts + shadow_imp_rel_counts
                shadow_dict[(act, scene)] = shadow_total_counts

            # Convert lowercase normalized names to original proper case and check if gathered names are in the list of play characters who participate
            # Note this primarily only works for plays as we have a convenient list of characters in the play and the worldbuilding does not cross this boundary.
            # However, longer more complex works like epic fantasy or scifi should include those nonpresent 3rd parties mentioned even if they do not actively participate. This is for worldbulding reasons.
            explicit_relation_counts = Counter({(original_character_map[a], original_character_map[b]): count for (a, b), count in explicit_relation_counts.items() if a in original_character_map and b in original_character_map})
            implicit_relation_counts = Counter({(original_character_map[a], original_character_map[b]): count for (a, b), count in implicit_relation_counts.items() if a in original_character_map and b in original_character_map})
            final_total_counts = explicit_relation_counts + implicit_relation_counts

            # Visualize character relations as signed weighted graph
            G = nx.DiGraph()

            for relation, weight in final_total_counts.items():
                G.add_edge(relation[0], relation[1], weight=weight)

            viz = Network(
                notebook=True, 
                cdn_resources="remote",
                neighborhood_highlight=True,
                select_menu=True,
                filter_menu=True,
                )
            for node in G.nodes():
                viz.add_node(node, label=node, size=G.degree(node)*2) # scale by 2 for node visualization
            viz.from_nx(G)
            viz.set_options("""
                {
                    "nodes": {
                        "font": {
                            "size": 50
                        }
                    },
                    "physics": {
                        "stabilization": {
                            "enabled": true,
                            "iterations": 300
                        },
                        "barnesHut": {
                            "gravitationalConstant": -80000,
                            "centralGravity": 0.3,
                            "springLength": 95,
                            "springConstant": 0.04
                        }
                    }
                }
                """)

            # Generate interactive static Pyvis graph visualizations, and export static graph to .graphml
            play_title = play[:play.find(".")]
            # viz.show(f"viz/{play_title}.html")
            nx.write_graphml(G, f"exports/graphml/{play_title}.graphml")

            # Export temporal graph snapshots to JSON files after converting shadow_dict to JSON-friendly format
            graph_snapshot = {
                str(tuple(int(t) for t in time_tuple)): {str(inner_k): inner_v for inner_k, inner_v in inner_relations.items()} for time_tuple, inner_relations in shadow_dict.items()
            }
            with open(f"exports/snapshots/{play_title} Temporal-Snapshot.json", "w") as f:
                json.dump(graph_snapshot, f, indent=4)

            end_time = time.time()
            elapsed_time = round(end_time - start_time, 1)
            cumulative_elapsed_time += elapsed_time
            tqdm.write(f"{play_title} done processing... ({elapsed_time}s)\n")
            pbar.update(1)

    tqdm.write(f"\nAll {len(plays)} plays are done processing. ({round(cumulative_elapsed_time, 1)}s)\n")