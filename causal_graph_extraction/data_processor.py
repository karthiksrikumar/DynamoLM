from entity_extractor import EntityExtractor
from causal_extractor import CausalExtractor
from temporal_graph import TemporalGraph
from heapq import heappush, heappop

class DataProcessor:
    def __init__(self, domain_keywords=None):
        self.entity_extractor = EntityExtractor(domain_keywords)
        self.causal_extractor = CausalExtractor()
        self.graph = TemporalGraph()
        self.priority_queue = []  # (priority, text, timestamp)

    def process_batch(self, texts, timestamps):
        # Prioritize texts with more entities
        for text, ts in zip(texts, timestamps):
            entity_count = len(self.entity_extractor.extract_entities(text))
            heappush(self.priority_queue, (-entity_count, text, ts))
        
        # Process in priority order
        while self.priority_queue:
            _, text, timestamp = heappop(self.priority_queue)
            entities = self.entity_extractor.extract_entities(text)
            causal_pairs = self.causal_extractor.extract_causal_relationships(text)
            for cause, effect, strength in causal_pairs:
                self.graph.add_causal_relationship(cause, effect, timestamp, strength=strength)
        self.graph.infer_indirect_relationships()

    def get_graph(self, time=None):
        return self.graph.get_graph_at_time(time) if time else self.graph.graph
