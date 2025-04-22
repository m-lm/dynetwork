import ast
import networkx as nx
import json

def recreate_graph(relations):
    G = nx.DiGraph()
    for rel, weight in relations.items():
        pair = ast.literal_eval(rel)
        G.add_edge(pair[0], pair[1], weight=weight)
    return G

def generate_visualization():
    play_title = "A Midsummer Night's Dream"
    with open(f"exports/snapshots/{play_title} Temporal-Snapshots.json") as f:
        snapshots = json.load(f)

    temporal_dict = {}
    for scene, relations in snapshots.items():
        temporal_dict[scene] = recreate_graph(relations)

    # TODO: need start and end attributes for gexf files to enable dynamic graphs in Gephi
    nx.write_gexf(G, f"exports/gexf/{play_title}.gexf")
    print(f"{play_title} exported to dynamic gexf")
    