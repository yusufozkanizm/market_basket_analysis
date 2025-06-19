"""
This code visualizes fptree created by fpgrowth algorithm
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import fpgrowth, association_rules

data = pd.read_csv(r"E:\piton\association\transaction_binary.csv")

frequent_itemsets = fpgrowth(data, min_support=0.05, use_colnames=True)

rules = association_rules(frequent_itemsets, metric='support', min_threshold=0.05)

G = nx.DiGraph()

node_sizes = {}
for _, row in frequent_itemsets.iterrows():
    items = list(row['itemsets'])
    support = row['support']

    for item in items:
        if item not in G:
            G.add_node(item, support=support)
            node_sizes[item] = support * 5000

edges = []
edge_weights = []
for _, row in rules.iterrows():
    antecedent = tuple(row['antecedents'])
    consequent = tuple(row['consequents'])
    support = row['support']

    for a in antecedent:
        for c in consequent:
            if a in G.nodes and c in G.nodes:
                G.add_edge(a, c, weight=support)
                edges.append((a, c))
                edge_weights.append(support * 10)

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)

nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=[node_sizes[node] for node in G.nodes()])

nx.draw_networkx_edges(G, pos, edgelist=edges, width=edge_weights, edge_color='gray')

labels = {node: node for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight="bold")

plt.title("FP-Tree with Individual Nodes for Each Product")

plt.savefig(r"E:\piton\association\fp_tree_visualized.png", format="PNG")

plt.show()
