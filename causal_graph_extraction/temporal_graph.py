from collections import defaultdict
import networkx as nx

class TemporalGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.edge_times = defaultdict(list)
        self.edge_strengths = defaultdict(float)

    def add_causal_relationship(self, cause, effect, start_time, end_time=None, strength=1.0):
        self.graph.add_edge(cause, effect)
        self.edge_times[(cause, effect)].append((start_time, end_time))
        self.edge_strengths[(cause, effect)] = max(self.edge_strengths[(cause, effect)], strength)

    def infer_indirect_relationships(self):
        indirect_edges = []
        for node in self.graph.nodes():
            for u, v in nx.dfs_edges(self.graph, source=node):
                if (u, v) not in self.edge_times and u != v:
                    strength = self.edge_strengths.get((u, v), 1.0) * 0.5
                    times = self.edge_times.get((u, v), [])
                    if times:
                        indirect_edges.append((u, v, times[0][0], times[0][1], strength))
        for u, v, start, end, strength in indirect_edges:
            self.add_causal_relationship(u, v, start, end, strength)

    def get_graph_at_time(self, t):
        subgraph = nx.DiGraph()
        for (u, v), intervals in self.edge_times.items():
            for start, end in intervals:
                if (end is None or t <= end) and t >= start:
                    subgraph.add_edge(u, v, strength=self.edge_strengths[(u, v)])
                    break
        return subgraph
