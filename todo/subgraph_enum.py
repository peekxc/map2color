import numpy as np
from scipy import sparse
from typing import Set, List, Optional, Union


class PivotKSubgraphEnumerator:
	"""
	Implementation of the Pivot algorithm for enumerating connected k-vertex subgraphs.
	Based on Algorithm 3 from "Enumerating Connected Induced Subgraphs: Improved Delay and Experimental Comparison"
	"""

	def __init__(self, adjacency_matrix: sparse.csr_matrix) -> None:
		"""
		Initialize with a scipy CSR sparse adjacency matrix.

		Args:
		    adjacency_matrix: scipy CSR sparse matrix representing the graph
		"""
		self.adj_matrix = adjacency_matrix.tocsr()
		self.n_nodes = self.adj_matrix.shape[0]

		# Precompute neighbor sets for efficiency
		self.neighbors: List[Set[int]] = []
		for i in range(self.n_nodes):
			row = self.adj_matrix.getrow(i)
			_, neighs = row.nonzero()
			self.neighbors.append(set(neighs.tolist()))

	def enumerate_k_connected_subgraphs(self, k: int) -> List[List[int]]:
		"""Enumerate all connected k-vertex subgraphs using the meta-algorithm with Pivot.

		Args:
		    k: size of subgraphs to enumerate

		Returns:
		    List of connected k-subgraphs (each as a sorted list of node indices)
		"""
		if k <= 0 or k > self.n_nodes:
			return []

		self.k = k
		all_results: List[Set[int]] = []

		# Meta-algorithm: while |V(G)| >= k
		active_vertices = set(range(self.n_nodes))

		while len(active_vertices) >= k:
			# Choose vertex v from V(G) - use lowest numbered for deterministic results
			v = min(active_vertices)

			# Enumerate all solutions containing v with Enum-Algo (Pivot)
			self.result: List[Set[int]] = []
			self._pivot(P={v}, S=set(), p=v, F=set(), excluded_vertices=set(range(self.n_nodes)) - active_vertices)

			# Add results to global collection
			all_results.extend(self.result)

			# Remove v from G (logically, not from the matrix)
			active_vertices.remove(v)

		# Convert to sorted lists and remove duplicates
		unique_subgraphs = []
		for subgraph in all_results:
			unique_subgraphs.append(tuple(sorted(subgraph)))

		return [list(subgraph) for subgraph in sorted(unique_subgraphs)]

	def _pivot(self, P: Set[int], S: Set[int], p: Optional[int], F: Set[int], excluded_vertices: Set[int] = None) -> None:
		"""The Pivot algorithm implementation following Algorithm 3.

		Args:
		    P: Current partial subgraph being built
		    S: Additional nodes in the subgraph
		    p: Current pivot node (can be None)
		    F: Forbidden nodes
		    excluded_vertices: Vertices to exclude (simulates removal from graph)
		"""
		if excluded_vertices is None:
			excluded_vertices = set()

		# Line 2-4: if |P ∪ S| = k then output P ∪ S and return
		if len(P | S) == self.k:
			self.result.append(P | S)
			return

		# Line 5-9: if p = null then choose pivot or return
		if p is None:
			available_P = P - excluded_vertices
			if available_P:
				p = next(iter(available_P))  # choose some element of available P
			else:
				return

		# Line 10-12: for z ∈ N(p) \ {P ∪ S ∪ F} (excluding removed vertices)
		neighbors_p = self.neighbors[p] - excluded_vertices
		excluded = P | S | F
		candidates = neighbors_p - excluded

		for z in candidates:
			self._pivot(P | {z}, S, p, F, excluded_vertices)
			F = F | {z}

		# Line 13: Pivot(P \ {p}, S ∪ {p}, null, F)
		self._pivot(P - {p}, S | {p}, None, F, excluded_vertices)

	def enumerate_simple_fwd(self, P: set, X: set, k: int):
		if len(P) == k:
			self.result.append(P)
		hasIntLeaf = True
		while not X:
			u = X.pop()
			# X |

		pass


def from_edgelist(edges: List[tuple], n_nodes: Optional[int] = None) -> sparse.csr_matrix:
	if not edges:
		return sparse.csr_matrix((0, 0))
	if n_nodes is None:
		n_nodes = max(max(edge) for edge in edges) + 1
	rows, cols = zip(*edges)
	data = [1] * len(edges)
	return sparse.csr_matrix((data + data, (rows + cols, cols + rows)), shape=(n_nodes, n_nodes))


# Example usage
if __name__ == "__main__":
	# Create a simple test graph: triangle with additional node
	# edges = [(0, 1), (1, 2), (1, 4), (2, 4), (3, 4)]
	edges = [(0, 1), (0, 9), (7, 9), (1, 9), (1, 5), (2, 5), (5, 8), (4, 5), (3, 4), (4, 6)]
	# edges = [(0, 1), (1, 2), (2, 0), (1, 3)]
	adj_matrix = create_graph_from_edges(edges, n_nodes=10)

	enumerator = PivotKSubgraphEnumerator(adj_matrix)

	print("Graph adjacency matrix:")
	print(adj_matrix.toarray())
	print()

	# Test with different k values
	for k in range(1, 5):
		subgraphs = enumerator.enumerate_k_connected_subgraphs(k)
		print(f"Connected {k}-vertex subgraphs: {len(subgraphs)}")
		for i, subgraph in enumerate(subgraphs):
			print(f"  {i+1}: {subgraph}")
		print()

	# Test on complete graph K4
	print("=" * 40)
	print("Complete graph K4:")
	k4_edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
	k4_matrix = create_graph_from_edges(k4_edges, n_nodes=4)
	k4_enumerator = PivotKSubgraphEnumerator(k4_matrix)

	k = 3
	k4_subgraphs = k4_enumerator.enumerate_k_connected_subgraphs(k)
	print(f"Connected {k}-vertex subgraphs in K4: {len(k4_subgraphs)}")
	for i, subgraph in enumerate(k4_subgraphs):
		print(f"  {i+1}: {subgraph}")
