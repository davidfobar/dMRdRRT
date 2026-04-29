# ──────────────────────────────────────────────────────────────────────────────
# File: graphkit/graphs.py
# ──────────────────────────────────────────────────────────────────────────────
import networkx as nx
from itertools import combinations
import random
import matplotlib.pyplot as plt

class SwarmGraph:
    def __init__(self, type: str, num_nodes: int, **kwargs):
        self.type = type
        self.num_nodes = num_nodes
        self.params = kwargs

    def add_adversarial_nodes(self, num_adversarial: int):
        self.params['num_adversarial'] = num_adversarial

    def build(self, adversarial_nodes: set = None):
        if self.type == "m_step_path":
            m = self.params.get('m', 1)
            self.graph = m_step_path_graph(self.num_nodes, m)

        if adversarial_nodes is not None:
            self.adversarial_nodes = adversarial_nodes
            self.params['num_adversarial'] = len(adversarial_nodes)

        elif 'num_adversarial' in self.params:
            self.adversarial_nodes = mark_adversaries(
                self.graph,
                self.params['num_adversarial'],
                seed=self.params.get('seed', None)
            )
        else:
            # No adversaries specified; default to empty set
            self.adversarial_nodes = set()

    def plot(self, remove_adversarial_edges: bool = False):
        if remove_adversarial_edges and 'num_adversarial' in self.params:
            is_conn, H, disabled_edges = connected_after_disabling_adversaries(
                self.graph, 
                adversarial_attr="adversarial"
            )
            graph_to_plot = H
        else:
            graph_to_plot = self.graph

        pos = nx.circular_layout(graph_to_plot)
        node_colors = ["red" if graph_to_plot.nodes[n].get("adversarial", False) else "blue" for n in graph_to_plot.nodes]
        nx.draw(graph_to_plot, pos, with_labels=True, node_color=node_colors, edge_color='black')
        plt.title(f"{self.type} Graph with {self.num_nodes} nodes")
        plt.show()

    def neighbors(self, i: int):
        return [int(j) for j in self.graph.neighbors(i) if j != i]

    def as_nx(self):
        return self.graph

    def degree(self, i: int):
        return int(self.as_nx().degree(i))

    def degrees(self):
        G = self.as_nx()
        return {int(n): int(d) for n, d in G.degree()}
    
    def get_adversarial_nodes(self):
        if 'num_adversarial' in self.params:
            return self.adversarial_nodes
        else:
            return set()

def m_step_path_graph(N: int, m: int) -> nx.Graph:
    """
    m-step path graph: connect i-j if their distance on a path is <= m.
    This is the m-th power of the path graph P_N.
    """
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(i+1, N):
            if (j - i) <= m:
                G.add_edge(i, j)
    return G

def connected_after_disabling_adversaries(
    G: nx.Graph,
    adversarial_attr: str = "adversarial"
):
    """
    Disables (removes) all edges incident to adversarial nodes, but
    keeps the adversarial nodes in the graph.

    Returns:
    - is_connected: connectivity of the NORMAL-agent-induced graph
    - H: graph with adversarial edges removed (nodes preserved)
    - disabled_edges: list of edges removed
    """
    H = G.copy()

    adversaries = [n for n, d in G.nodes(data=True) if d.get(adversarial_attr, False)]

    disabled_edges = []
    for a in adversaries:
        for u in list(H.neighbors(a)):
            disabled_edges.append((a, u))
            H.remove_edge(a, u)

    # Check connectivity among non-adversarial nodes
    normal_nodes = [n for n in H.nodes if n not in adversaries]
    N_sub = H.subgraph(normal_nodes)

    if N_sub.number_of_nodes() == 0:
        return False, H, disabled_edges

    if H.is_directed():
        is_conn = nx.is_weakly_connected(N_sub)
    else:
        is_conn = nx.is_connected(N_sub)

    return is_conn, H, disabled_edges

def mark_adversaries(G: nx.Graph, x: int, seed = None):
    if x == 0:
        adv = set()
        for n in G.nodes:
            G.nodes[n]["adversarial"] = False
        return adv
    
    if seed is None:
        rng = random.Random()
    else:
        rng = random.Random(seed)
    adv = set(rng.sample(list(G.nodes), x))
    for n in G.nodes:
        G.nodes[n]["adversarial"] = (n in adv)
    return adv

def r_robustness(self) -> int:
    """
    Compute the r-robustness of the current graph.
    
    A graph is r-robust if for every pair of non-empty subsets S1, S2
    of V, at least one subset has a node with >= r neighbors outside
    of that subset.
    
    Returns the maximum r for which the graph is r-robust.
    Uses global knowledge — not suitable for decentralized use.
    """
    G = self.graph
    nodes = list(G.nodes)
    N = len(nodes)
    
    if N == 0:
        return 0

    def is_r_robust(r: int) -> bool:
        # Check all pairs of non-empty disjoint subsets S1, S2
        for size1 in range(1, N):
            for size2 in range(1, N - size1 + 1):
                for S1 in combinations(nodes, size1):
                    S1 = set(S1)
                    remaining = [n for n in nodes if n not in S1]
                    for S2 in combinations(remaining, size2):
                        S2 = set(S2)
                        # At least one subset must have a node
                        # with >= r neighbors outside that subset
                        S1_satisfies = any(
                            sum(1 for nb in G.neighbors(v) if nb not in S1) >= r
                            for v in S1
                        )
                        S2_satisfies = any(
                            sum(1 for nb in G.neighbors(v) if nb not in S2) >= r
                            for v in S2
                        )
                        if not S1_satisfies and not S2_satisfies:
                            return False
        return True

    # Binary search over r
    lo, hi = 0, N // 2
    result = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if is_r_robust(mid):
            result = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return result