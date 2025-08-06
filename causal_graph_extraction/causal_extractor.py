import spacy

class CausalExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.causal_verbs = {"cause", "lead", "result", "affect", "influence"}
        self.causal_phrases = {"due to", "because of", "leads to"}

    def extract_causal_relationships(self, text):
        doc = self.nlp(text)
        causal_pairs = []
        
        # Pattern matching and dependency parsing
        for sent in doc.sents:
            for token in sent:
                if token.lemma_ in self.causal_verbs and token.dep_ == "ROOT":
                    subject = [child for child in token.children if child.dep_ == "nsubj"]
                    object_ = [child for child in token.children if child.dep_ in {"dobj", "prep"}]
                    if subject and object_:
                        strength = 1.0 if token.text in {"cause", "leads"} else 0.7
                        causal_pairs.append((subject[0].text, object_[0].text, strength))
                
                # Phrase-based detection
                phrase = " ".join(t.text for t in sent).lower()
                for cp in self.causal_phrases:
                    if cp in phrase:
                        parts = phrase.split(cp)
                        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                            causal_pairs.append((parts[0].strip(), parts[1].strip(), 0.5))
        
        return causal_pairs
