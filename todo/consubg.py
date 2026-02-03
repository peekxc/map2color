from typing import Generator
from networkx import Graph
import numpy as np
from collections import defaultdict
from itertools import combinations
from functools import lru_cache, cache, reduce as freduce


def compositions(n: int, k: int) -> Generator:
	if n < 0 or k < 0 or (k == 0 and n == 0):
		yield []
	elif k == 1:
		yield [n]
	else:
		for i in range(1, n):
			for comp in compositions(n - i, k - 1):
				yield [i, *comp]


class GraphSubgEnumerator:
	"""Enumerater class for all connected subgraphs of a specific size from a graph.

	Based on the algorithm from Karakashian et al. (2013).
	"""

	def __init__(self, graph):
		"""Initialize with a graph represented as a dictionary of sets."""
		self.G = graph
		self.vtx_to_idx = {v: i for i, v in enumerate(self.G.keys())}
		self.idx_to_vtx = {i: v for v, i in self.vtx_to_idx.items()}

	def get_conn_subgraphs(self, k, symmetric=False):
		"""Returns all connected subgraphs of size k from the graph.

		Parameters:
			k: Size of subgraphs to enumerate
			symmetric: If True, avoids generating isomorphic subgraphs

		Returns:
			set of frozensets, where each frozenset is a connected subgraph of size k
		"""
		all_sgs = set()

		if symmetric:
			# Create a copy of the graph that we'll progressively reduce
			rem_G = dict(self.G)
			for u in self.G.keys():
				# Get combinations with vertex u
				all_sgs |= self._combs_with_vtx(u, k, rem_G)

				# Remove vertex u from the remaining graph
				rem_G = {v: v_neighbors - {u} for v, v_neighbors in rem_G.items() if v != u}
		else:
			# Process all vertices without removing any
			for u in self.G.keys():
				all_sgs |= self._combs_with_vtx(u, k, self.G)

		return all_sgs

	def _combs_with_vtx(self, u: int, k: int, graph: dict):
		"""Find all connected subgraphs of size k that include vertex u."""
		comb_tree = CombTreeBuilder(u, k, graph)
		return comb_tree.combinations()


class CombTreeBuilder:
	"""Builds a tree representing combinations of vertices connected to the root vertex."""

	def __init__(self, root: int, k: int, graph: dict):
		"""Initialize with a root vertex, size k, and the graph."""
		self.root = root
		self.k = k
		self.G = graph
		self.tree = defaultdict(set)
		self.tl = {0: root}  # Maps tree node IDs to graph vertices

		# Set up numpy arrays for marking nodes and vertices
		n_vtx = len(graph)
		self.vtx_to_idx = {v: i for i, v in enumerate(graph.keys())}
		self.idx_to_vtx = {i: v for v, i in self.vtx_to_idx.items()}
		self.mark_vtx = np.zeros(n_vtx, dtype=bool)
		self.mark_node = dict()
		self.build_tree()

	def build_tree(self):
		"""Build a tree representing the structure of the graph from the root vertex."""
		self.levels = [{self.root}] + [set() for _ in range(1, self.k)]
		self._build_tree(0, 1, self.k)

	def _build_tree(self, node_id, depth, k):
		"""Recursively build the tree structure.

		Parameters:
			node_id: ID of the current node in the tree
			depth: Current depth in the tree
			k: Maximum depth to build
		"""
		self.levels[depth] = set(self.levels[depth - 1])
		curr_vtx = self.tl[node_id]

		for v in self.G[curr_vtx]:
			if v != node_id and v not in self.levels[depth]:
				new_id = len(self.tl)
				self.tl[new_id] = v
				self.tree[node_id].add(new_id)
				self.levels[depth].add(v)

				# Mark the node if the vertex hasn't been seen before
				v_idx = self.vtx_to_idx[v]
				if not self.mark_vtx[v_idx]:
					self.mark_node[new_id] = True
					self.mark_vtx[v_idx] = True
				else:
					self.mark_node[new_id] = False

				# Continue building the tree if we haven't reached the maximum depth
				if depth + 1 <= k - 1:
					self._build_tree(new_id, depth + 1, k)

	def combinations(self):
		"""Get all connected subgraphs of size k that include the root vertex."""
		result = {frozenset({self.tl[i] for i in fs}) for fs in self.combs_from_tree(0, self.k)}
		return result

	@lru_cache(maxsize=None)
	def combs_from_tree(self, root: int, k: int):
		"""Recursively enumerate all combinations of nodes in the tree.

		Parameters:
			root: ID of the root node in the tree
			k: Size of combinations to generate

		Returns:
			Set of frozensets, where each frozenset is a combination of tree node IDs
		"""
		## Base case: k=1 means just the root node
		if k == 1:
			return {frozenset({root})}

		node_sets = set()
		MAX_SIZE = min(len(self.tree[root]), k - 1) + 1
		for i in range(1, MAX_SIZE):
			for node_comb in combinations(self.tree[root], i):
				for dist in compositions(k - 1, i):
					# Get combinations for each child according to the distribution
					sub_combs = [self.combs_from_tree(ch, sz) for ch, sz in zip(node_comb[:i], dist[:i])]

					# Combine the sub-combinations
					node_sets.update({frozenset({root} | combined) for combined in self.union_prod(sub_combs)})

		return node_sets

	def union_prod(self, comb_sets: list[set]) -> set:
		"""Compute the union product of sets of combinations, enforcing constraints.

		Parameters:
			comb_sets: List of sets, where each set contains combinations

		Returns:
			Set of combinations that satisfy the constraints
		"""
		if not comb_sets:
			return set()

		if len(comb_sets) == 1:
			return comb_sets[0]

		result = set()
		for s1 in comb_sets[0]:
			for s2 in self.union_prod(comb_sets[1:]):
				# Convert tree node IDs to graph vertices for checking
				s1_vtx = {self.tl[i] for i in s1}
				s2_vtx = {self.tl[j] for j in s2}

				# Check if the sets are disjoint
				if s1_vtx.isdisjoint(s2_vtx):
					# Constraint: either some node in s2 is marked, or no vertex in s1 has a neighbor in s2
					s2_marked = (self.mark_node.get(j, False) for j in s2)
					it_disjoint = ({self.tl[j] for j in self.tree[i]}.isdisjoint(s2_vtx) for i in s1)
					if any(s2_marked) or all(it_disjoint):
						result.add(s1 | s2)

		return result


# Example usage
if __name__ == "__main__":
	import networkx as nx

	G = nx.watts_strogatz_graph(n=40, k=4, p=0.20)
	L = {n: set(nx.neighbors(G, n)) for n in G.nodes}
	coords = np.array(list(nx.forceatlas2_layout(G).values()))

	from bokeh.io import output_notebook
	from bokeh.plotting import figure, show

	output_notebook()
	p = figure(width=350, height=350)
	p.scatter(*coords.T, size=15)

	xs = [coords[e, 0] for e in G.edges]
	ys = [coords[e, 1] for e in G.edges]
	p.multi_line(xs, ys)

	p.text(*coords.T, text=np.arange(len(coords)))
	show(p)

	# Create enumerator and get subgraphs of size 3
	enumerator = GraphSubgEnumerator(L)
	subgraphs = enumerator.get_conn_subgraphs(5, symmetric=True)
	print(f"Found {len(subgraphs)} connected subgraphs:")
	for sg in subgraphs:
		cc = list(nx.connected_components(nx.subgraph(G, sg)))
		assert len(cc) == 1, f"Subgraph {sg} not connected"
