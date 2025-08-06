import spacy
from collections import defaultdict

class EntityExtractor:
    def __init__(self, domain_keywords=None):
        self.nlp = spacy.load("en_core_web_sm")
        self.domain_keywords = domain_keywords or set()
        self.entity_memory = defaultdict(int)  # Feedback loop storage

    def extract_entities(self, text):
        doc = self.nlp(text)
        entities = set()
        
        # Base NER
        for ent in doc.ents:
            entities.add((ent.text, ent.label_))
        
        # Domain-specific enhancement
        for token in doc:
            if token.text.lower() in self.domain_keywords:
                entities.add((token.text, "DOMAIN"))
        
        # Contextual disambiguation
        for i, token in enumerate(doc):
            if token.text in self.entity_memory:
                context = " ".join(t.text for t in doc[max(0, i-2):i+3])
                entities.add((f"{token.text}_{context[:10]}", "CUSTOM"))
        
        # Update memory
        for ent, _ in entities:
            self.entity_memory[ent] += 1
        
        return list(entities)
