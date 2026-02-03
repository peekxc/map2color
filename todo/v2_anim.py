# %%
import numpy as np
import networkx as nx
from bokeh.plotting import figure, show
from bokeh.io import show, output_notebook

output_notebook()

# %% Candidate pair animation
GN = nx.watts_strogatz_graph(n=10, k=4, p=0.55)
HN = nx.watts_strogatz_graph(n=5, k=3, p=0.50)

matcher = nx.isomorphism.GraphMatcher(G, H)


# for s in matcher.subgraph_isomorphisms_iter():
# from vf2 import VF2
G = Graph()
for n in GN.nodes:
	G.add_node(n)
for i, j in GN.edges:
	G.add_edge(i, j)

H = Graph()
for n in HN.nodes:
	H.add_node(n)
for i, j in HN.edges:
	H.add_edge(i, j)

vf2 = VF2(H, G)
vf2.find_subgraph_isomorphisms()

# %%


# %%
# from hirola import HashTable
# class Matching:
# 	def __init__(self, mapping: list[tuple], n: int) -> None:
# 		np.
# 		self.
