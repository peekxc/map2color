import numpy as np
from collections import defaultdict
from itertools import product
from typing import Dict, List, Set, Tuple, Optional, Any, Callable


class Graph:
	def __init__(self):
		self.nodes = set()
		self.edges = defaultdict(set)
		self.node_attrs = {}
		self.edge_attrs = {}

	def add_node(self, node: Any, attr: Dict = None) -> None:
		self.nodes.add(node)
		if attr:
			self.node_attrs[node] = attr

	def add_edge(self, u: Any, v: Any, attr: Dict = None) -> None:
		self.nodes.add(u)
		self.nodes.add(v)
		self.edges[u].add(v)
		self.edges[v].add(u)
		if attr:
			self.edge_attrs[(u, v)] = attr
			self.edge_attrs[(v, u)] = attr

	def neighbors(self, node: Any) -> Set:
		return self.edges[node]


class VF2:
	"""VF2 algorithm for graph and subgraph isomorphism."""

	def __init__(self, pattern: Graph, target: Graph):
		"""Initialize VF2 algorithm."""
		self.h = pattern
		self.g = target
		self.reset_state()

	def reset_state(self) -> None:
		"""Reset the search state."""
		self.core_h = {}
		self.core_g = {}
		self.in_h = set()
		self.in_g = set()
		self.out_h = set()
		self.out_g = set()
		self.unmapped_h = set(self.h.nodes)
		self.unmapped_g = set(self.g.nodes)

	def is_isomorphic(self) -> bool:
		"""Check if pattern is isomorphic to target."""
		if len(self.h.nodes) != len(self.g.nodes):
			return False
		if sum(len(adj) for adj in self.h.edges.values()) != sum(len(adj) for adj in self.g.edges.values()):
			return False
		self.reset_state()
		return self._match()

	def subgraph_isomorphic(self) -> bool:
		"""Check if pattern is subgraph isomorphic to target."""
		if len(self.h.nodes) > len(self.g.nodes):
			return False
		self.reset_state()
		return self._match()

	def find_isomorphisms(self) -> List[Dict]:
		"""Find all isomorphisms between pattern and target."""
		if len(self.h.nodes) != len(self.g.nodes):
			return []
		self.reset_state()
		isomorphisms = []
		self._match_all(isomorphisms)
		return isomorphisms

	def find_subgraph_isomorphisms(self) -> List[Dict]:
		"""Find all subgraph isomorphisms of pattern in target."""
		if len(self.h.nodes) > len(self.g.nodes):
			return []
		self.reset_state()
		isomorphisms = []
		self._match_all(isomorphisms)
		return isomorphisms

	def _match(self) -> bool:
		"""Find a single isomorphism."""
		if len(self.core_h) == len(self.h.nodes):
			return True

		nh = next(iter(self.out_h)) if self.out_h else next(iter(self.unmapped_h), None)
		if nh is None:
			return True  # For subgraph isomorphism

		candidates = self.out_g.copy() if self.out_h else self.unmapped_g.copy()
		for ng in candidates:
			if self._is_feasible(nh, ng):
				self._add_mapping(nh, ng)
				if self._match():
					return True
				self._remove_mapping(nh, ng)
		return False

	def _match_all(self, isomorphisms: List[Dict]) -> None:
		"""Find all isomorphisms."""
		if len(self.core_h) == len(self.h.nodes):
			isomorphisms.append(self.core_h.copy())
			return

		nh = next(iter(self.out_h)) if self.out_h else next(iter(self.unmapped_h), None)
		if nh is None:
			isomorphisms.append(self.core_h.copy())
			return

		candidates = self.out_g.copy() if self.out_h else self.unmapped_g.copy()
		for ng in candidates:
			if self._is_feasible(nh, ng):
				self._add_mapping(nh, ng)
				self._match_all(isomorphisms)
				self._remove_mapping(nh, ng)

	def _is_feasible(self, nh: Any, ng: Any) -> bool:
		"""Check if mapping nh->ng is feasible."""
		feasibility_checks = [self._check_node_attrs(nh, ng), self._check_edges(nh, ng)]
		return all(feasibility_checks)

	def _check_node_attrs(self, nh: Any, ng: Any) -> bool:
		"""Check node attribute compatibility."""
		if nh in self.h.node_attrs and ng in self.g.node_attrs:
			return self.h.node_attrs[nh] == self.g.node_attrs[ng]
		return True

	def _check_edges(self, nh: Any, ng: Any) -> bool:
		"""Check edge compatibility."""
		return all(
			self._check_edge_exists(nh, ng, curr_h, curr_g) and self._check_edge_attrs(nh, ng, curr_h, curr_g)
			for curr_h, curr_g in self.core_h.items()
		)

	def _check_edge_exists(self, nh: Any, ng: Any, curr_h: Any, curr_g: Any) -> bool:
		"""Check if edge existence is consistent."""
		edge_h_exists = nh in self.h.neighbors(curr_h)
		edge_g_exists = ng in self.g.neighbors(curr_g)
		return edge_h_exists == edge_g_exists

	def _check_edge_attrs(self, nh: Any, ng: Any, curr_h: Any, curr_g: Any) -> bool:
		"""Check edge attribute compatibility."""
		edge_h_exists = nh in self.h.neighbors(curr_h)
		edge_g_exists = ng in self.g.neighbors(curr_g)

		if not (edge_h_exists and edge_g_exists):
			return True

		if (curr_h, nh) in self.h.edge_attrs and (curr_g, ng) in self.g.edge_attrs:
			return self.h.edge_attrs[(curr_h, nh)] == self.g.edge_attrs[(curr_g, ng)]
		return True

	def _add_mapping(self, nh: Any, ng: Any) -> None:
		"""Add mapping nh->ng to state."""
		self.core_h[nh] = ng
		self.core_g[ng] = nh

		self._remove_from_sets(nh, self.out_h)
		self._remove_from_sets(ng, self.out_g)
		self._remove_from_sets(nh, self.unmapped_h)
		self._remove_from_sets(ng, self.unmapped_g)

		self._update_in_out_sets(nh, ng)

	def _remove_from_sets(self, node: Any, node_set: Set) -> None:
		"""Remove node from set if present."""
		if node in node_set:
			node_set.remove(node)

	def _update_in_out_sets(self, nh: Any, ng: Any) -> None:
		"""Update in/out sets after adding a mapping."""
		for h_neigh in self.h.neighbors(nh):
			if h_neigh not in self.core_h:
				self._remove_from_sets(h_neigh, self.in_h)
				if h_neigh not in self.out_h:
					self.out_h.add(h_neigh)
				self._remove_from_sets(h_neigh, self.unmapped_h)

		for g_neigh in self.g.neighbors(ng):
			if g_neigh not in self.core_g:
				self._remove_from_sets(g_neigh, self.in_g)
				if g_neigh not in self.out_g:
					self.out_g.add(g_neigh)
				self._remove_from_sets(g_neigh, self.unmapped_g)

	def _remove_mapping(self, nh: Any, ng: Any) -> None:
		"""Remove mapping nh->ng from state."""
		del self.core_h[nh]
		del self.core_g[ng]

		# Rebuild state from scratch
		self.unmapped_h = set(self.h.nodes) - set(self.core_h.keys())
		self.unmapped_g = set(self.g.nodes) - set(self.core_g.keys())
		self.in_h = set()
		self.in_g = set()
		self.out_h = set()
		self.out_g = set()

		# Update in/out sets based on current mappings
		for h, g in self.core_h.items():
			self._update_neighbors_after_mapping(h, g)

	def _update_neighbors_after_mapping(self, h: Any, g: Any) -> None:
		"""Update neighbor sets after mapping."""
		for h_neigh in self.h.neighbors(h):
			if h_neigh not in self.core_h:
				self._remove_from_sets(h_neigh, self.unmapped_h)
				if any(n in self.core_h for n in self.h.neighbors(h_neigh)):
					if h_neigh not in self.out_h:
						self.out_h.add(h_neigh)
				else:
					if h_neigh not in self.in_h:
						self.in_h.add(h_neigh)

		for g_neigh in self.g.neighbors(g):
			if g_neigh not in self.core_g:
				self._remove_from_sets(g_neigh, self.unmapped_g)
				if any(n in self.core_g for n in self.g.neighbors(g_neigh)):
					if g_neigh not in self.out_g:
						self.out_g.add(g_neigh)
				else:
					if g_neigh not in self.in_g:
						self.in_g.add(g_neigh)
