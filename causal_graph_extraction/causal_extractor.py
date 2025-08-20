import spacy
import re
from collections import defaultdict
from typing import List, Tuple, Dict, Set
import numpy as np
from dataclasses import dataclass

@dataclass
class CausalRelation:
    cause: str
    effect: str
    strength: float
    relation_type: str
    evidence_span: Tuple[int, int]
    confidence: float

class ImprovedCausalExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        
        # Expanded causal indicators with weights
        self.causal_verbs = {
            "cause": 0.95, "causes": 0.95, "caused": 0.95,
            "lead": 0.85, "leads": 0.85, "led": 0.85,
            "result": 0.80, "results": 0.80, "resulted": 0.80,
            "trigger": 0.90, "triggers": 0.90, "triggered": 0.90,
            "generate": 0.75, "generates": 0.75, "generated": 0.75,
            "produce": 0.75, "produces": 0.75, "produced": 0.75,
            "induce": 0.85, "induces": 0.85, "induced": 0.85,
            "create": 0.70, "creates": 0.70, "created": 0.70,
            "bring": 0.65, "brings": 0.65, "brought": 0.65,
            "contribute": 0.60, "contributes": 0.60, "contributed": 0.60,
            "influence": 0.65, "influences": 0.65, "influenced": 0.65,
            "affect": 0.70, "affects": 0.70, "affected": 0.70,
            "enable": 0.75, "enables": 0.75, "enabled": 0.75,
            "force": 0.80, "forces": 0.80, "forced": 0.80
        }
        
        self.causal_phrases = {
            "due to": 0.85,
            "because of": 0.90,
            "as a result of": 0.95,
            "resulting from": 0.90,
            "owing to": 0.80,
            "thanks to": 0.75,
            "attributed to": 0.85,
            "stems from": 0.80,
            "arises from": 0.80,
            "derives from": 0.75,
            "originates from": 0.80,
            "leads to": 0.85,
            "results in": 0.90,
            "gives rise to": 0.85,
            "brings about": 0.80,
            "accounts for": 0.75,
            "is responsible for": 0.85,
            "contributes to": 0.70,
            "plays a role in": 0.65,
            "has an impact on": 0.70,
            "influences": 0.70,
            "triggers": 0.85,
            "prompts": 0.75,
            "sparks": 0.80,
            "facilitates": 0.75,
            "enables": 0.75
        }
        
        # Conditional indicators
        self.conditional_patterns = {
            "if.*then": 0.70,
            "when.*then": 0.75,
            "whenever": 0.70,
            "provided that": 0.65,
            "given that": 0.70,
            "assuming": 0.60,
            "in case": 0.65
        }
        
        # Negation words that might indicate inverse causation
        self.negations = {"not", "no", "never", "without", "lack", "absence", "prevents", "stops", "blocks"}
        
        # Entity types that are likely to be causes or effects
        self.causal_entities = {"ORG", "PERSON", "EVENT", "PRODUCT", "GPE", "NORP"}
        
    def extract_causal_relationships(self, text: str) -> List[CausalRelation]:
        doc = self.nlp(text)
        relations = []
        
        # Extract different types of causal patterns
        relations.extend(self._extract_verb_based_causation(doc))
        relations.extend(self._extract_phrase_based_causation(doc))
        relations.extend(self._extract_dependency_patterns(doc))
        relations.extend(self._extract_conditional_patterns(doc))
        relations.extend(self._extract_sequential_patterns(doc))
        
        # Post-process and filter
        relations = self._deduplicate_relations(relations)
        relations = self._filter_by_confidence(relations, threshold=0.3)
        relations = self._adjust_for_context(relations, doc)
        
        return relations
    
    def _extract_verb_based_causation(self, doc) -> List[CausalRelation]:
        relations = []
        
        for sent in doc.sents:
            for token in sent:
                if token.lemma_.lower() in self.causal_verbs:
                    base_strength = self.causal_verbs[token.lemma_.lower()]
                    
                    # Find subject (cause) and object (effect)
                    subject = self._find_subject(token)
                    obj = self._find_object(token)
                    
                    if subject and obj:
                        # Adjust strength based on context
                        strength = self._adjust_strength_for_context(
                            base_strength, token, sent
                        )
                        
                        confidence = self._calculate_confidence(subject, obj, token)
                        
                        relations.append(CausalRelation(
                            cause=subject.text,
                            effect=obj.text,
                            strength=strength,
                            relation_type="verb_based",
                            evidence_span=(token.idx, token.idx + len(token.text)),
                            confidence=confidence
                        ))
        
        return relations
    
    def _extract_phrase_based_causation(self, doc) -> List[CausalRelation]:
        relations = []
        text = doc.text.lower()
        
        for phrase, base_strength in self.causal_phrases.items():
            pattern = re.escape(phrase)
            matches = list(re.finditer(pattern, text))
            
            for match in matches:
                start, end = match.span()
                
                # Find entities before and after the phrase
                cause_entities = self._find_nearby_entities(doc, start, direction="before")
                effect_entities = self._find_nearby_entities(doc, end, direction="after")
                
                for cause in cause_entities[:2]:  # Limit to 2 nearest entities
                    for effect in effect_entities[:2]:
                        if cause != effect:
                            confidence = self._calculate_phrase_confidence(
                                cause, effect, phrase, doc
                            )
                            
                            relations.append(CausalRelation(
                                cause=cause.text,
                                effect=effect.text,
                                strength=base_strength,
                                relation_type="phrase_based",
                                evidence_span=(start, end),
                                confidence=confidence
                            ))
        
        return relations
    
    def _extract_dependency_patterns(self, doc) -> List[CausalRelation]:
        relations = []
        
        for sent in doc.sents:
            for token in sent:
                # Pattern 1: nsubj -> VERB -> dobj/pobj
                if token.dep_ == "ROOT" and token.pos_ == "VERB":
                    subjects = [child for child in token.children if child.dep_ in ["nsubj", "nsubjpass"]]
                    objects = [child for child in token.children if child.dep_ in ["dobj", "pobj", "attr"]]
                    
                    if subjects and objects:
                        for subj in subjects:
                            for obj in objects:
                                strength = self._get_verb_causality_score(token.lemma_)
                                if strength > 0.4:  # Only include likely causal verbs
                                    confidence = self._calculate_dependency_confidence(subj, obj, token)
                                    
                                    relations.append(CausalRelation(
                                        cause=subj.text,
                                        effect=obj.text,
                                        strength=strength,
                                        relation_type="dependency",
                                        evidence_span=(token.idx, token.idx + len(token.text)),
                                        confidence=confidence
                                    ))
                
                # Pattern 2: Passive constructions
                if token.dep_ == "nsubjpass":
                    agent = self._find_agent_in_passive(token.head)
                    if agent:
                        strength = self._get_verb_causality_score(token.head.lemma_)
                        if strength > 0.4:
                            confidence = self._calculate_dependency_confidence(agent, token, token.head)
                            
                            relations.append(CausalRelation(
                                cause=agent.text,
                                effect=token.text,
                                strength=strength * 0.9,  # Slightly lower confidence for passive
                                relation_type="passive_dependency",
                                evidence_span=(token.head.idx, token.head.idx + len(token.head.text)),
                                confidence=confidence
                            ))
        
        return relations
    
    def _extract_conditional_patterns(self, doc) -> List[CausalRelation]:
        relations = []
        text = doc.text.lower()
        
        for pattern, strength in self.conditional_patterns.items():
            matches = list(re.finditer(pattern, text))
            
            for match in matches:
                # Find entities in condition and consequence
                condition_entities = self._find_entities_in_span(doc, match.start(), match.end())
                # Look for entities after the conditional
                consequence_entities = self._find_nearby_entities(doc, match.end(), direction="after")
                
                for cond in condition_entities:
                    for cons in consequence_entities[:2]:
                        confidence = min(0.8, strength + 0.1)  # Conditionals are inherently less certain
                        
                        relations.append(CausalRelation(
                            cause=cond.text,
                            effect=cons.text,
                            strength=strength,
                            relation_type="conditional",
                            evidence_span=match.span(),
                            confidence=confidence
                        ))
        
        return relations
    
    def _extract_sequential_patterns(self, doc) -> List[CausalRelation]:
        """Extract temporal sequences that might imply causation"""
        relations = []
        
        # Look for temporal markers followed by events
        temporal_markers = ["after", "following", "subsequently", "then", "next", "later", "eventually"]
        
        for sent in doc.sents:
            sent_text = sent.text.lower()
            for marker in temporal_markers:
                if marker in sent_text:
                    # Find entities before and after temporal marker
                    marker_pos = sent_text.find(marker)
                    
                    before_entities = self._find_entities_in_span(doc, sent.start_char, sent.start_char + marker_pos)
                    after_entities = self._find_entities_in_span(doc, sent.start_char + marker_pos, sent.end_char)
                    
                    for before in before_entities:
                        for after in after_entities:
                            # Temporal sequence suggests possible causation but with lower confidence
                            relations.append(CausalRelation(
                                cause=before.text,
                                effect=after.text,
                                strength=0.4,  # Lower strength for temporal-only evidence
                                relation_type="temporal_sequence",
                                evidence_span=(sent.start_char + marker_pos, sent.start_char + marker_pos + len(marker)),
                                confidence=0.5
                            ))
        
        return relations
    
    def _find_subject(self, verb_token):
        """Find the subject of a verb token"""
        for child in verb_token.children:
            if child.dep_ in ["nsubj", "nsubjpass"]:
                return child
        return None
    
    def _find_object(self, verb_token):
        """Find the object of a verb token"""
        for child in verb_token.children:
            if child.dep_ in ["dobj", "pobj", "attr"]:
                return child
        return None
    
    def _find_agent_in_passive(self, verb_token):
        """Find the agent in passive constructions"""
        for child in verb_token.children:
            if child.dep_ == "agent" or (child.dep_ == "prep" and child.text.lower() == "by"):
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        return grandchild
        return None
    
    def _find_nearby_entities(self, doc, position: int, direction: str, max_distance: int = 100):
        """Find entities near a given position"""
        entities = []
        
        if direction == "before":
            search_start = max(0, position - max_distance)
            search_end = position
        else:  # after
            search_start = position
            search_end = min(len(doc.text), position + max_distance)
        
        for ent in doc.ents:
            if (search_start <= ent.start_char <= search_end and 
                ent.label_ in self.causal_entities):
                entities.append(ent)
        
        # Sort by distance from position
        entities.sort(key=lambda e: abs(e.start_char - position))
        return entities
    
    def _find_entities_in_span(self, doc, start: int, end: int):
        """Find entities within a specific span"""
        entities = []
        for ent in doc.ents:
            if start <= ent.start_char <= end and ent.label_ in self.causal_entities:
                entities.append(ent)
        return entities
    
    def _adjust_strength_for_context(self, base_strength: float, token, sent) -> float:
        """Adjust strength based on context clues"""
        strength = base_strength
        
        # Check for negations
        if any(neg in sent.text.lower() for neg in self.negations):
            strength *= 0.7
        
        # Check for uncertainty markers
        uncertainty_markers = ["might", "could", "possibly", "perhaps", "maybe"]
        if any(marker in sent.text.lower() for marker in uncertainty_markers):
            strength *= 0.8
        
        # Check for emphasis
        emphasis_markers = ["clearly", "definitely", "obviously", "certainly"]
        if any(marker in sent.text.lower() for marker in emphasis_markers):
            strength = min(1.0, strength * 1.1)
        
        return strength
    
    def _calculate_confidence(self, subject, obj, verb) -> float:
        """Calculate confidence based on entity types and context"""
        base_confidence = 0.7
        
        # Higher confidence if both are named entities
        if subject.ent_type_ and obj.ent_type_:
            base_confidence += 0.1
        
        # Higher confidence for certain entity combinations
        if (subject.ent_type_ in ["PERSON", "ORG"] and 
            obj.ent_type_ in ["EVENT", "PRODUCT"]):
            base_confidence += 0.1
        
        # Lower confidence for very common words
        if subject.text.lower() in ["it", "this", "that", "they"]:
            base_confidence -= 0.2
        
        return min(1.0, base_confidence)
    
    def _calculate_phrase_confidence(self, cause, effect, phrase, doc) -> float:
        """Calculate confidence for phrase-based extractions"""
        base_confidence = 0.6
        
        # Distance penalty - closer entities are more likely related
        distance = abs(cause.start_char - effect.start_char)
        if distance < 50:
            base_confidence += 0.2
        elif distance > 200:
            base_confidence -= 0.2
        
        return max(0.1, min(1.0, base_confidence))
    
    def _calculate_dependency_confidence(self, cause, effect, verb) -> float:
        """Calculate confidence for dependency-based extractions"""
        base_confidence = 0.8  # Higher for syntactic patterns
        
        # Check if entities make semantic sense
        if self._entities_semantically_compatible(cause.text, effect.text):
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _entities_semantically_compatible(self, cause: str, effect: str) -> bool:
        """Simple semantic compatibility check"""
        # This could be improved with word embeddings or semantic knowledge
        cause_lower = cause.lower()
        effect_lower = effect.lower()
        
        # Basic heuristics
        if cause_lower == effect_lower:
            return False
        
        # Check for obvious mismatches (this is very basic)
        abstract_concepts = {"love", "hate", "beauty", "truth", "justice"}
        concrete_objects = {"car", "house", "tree", "rock", "book"}
        
        cause_abstract = any(concept in cause_lower for concept in abstract_concepts)
        effect_concrete = any(obj in effect_lower for obj in concrete_objects)
        
        # Abstract causes with concrete effects are often valid
        if cause_abstract and effect_concrete:
            return False
        
        return True  # Default to compatible
    
    def _get_verb_causality_score(self, verb_lemma: str) -> float:
        """Get causality score for a verb"""
        return self.causal_verbs.get(verb_lemma.lower(), 0.0)
    
    def _deduplicate_relations(self, relations: List[CausalRelation]) -> List[CausalRelation]:
        """Remove duplicate relations, keeping the highest confidence ones"""
        seen = {}
        for relation in relations:
            key = (relation.cause.lower().strip(), relation.effect.lower().strip())
            if key not in seen or seen[key].confidence < relation.confidence:
                seen[key] = relation
        
        return list(seen.values())
    
    def _filter_by_confidence(self, relations: List[CausalRelation], threshold: float = 0.3) -> List[CausalRelation]:
        """Filter relations by confidence threshold"""
        return [r for r in relations if r.confidence >= threshold]
    
    def _adjust_for_context(self, relations: List[CausalRelation], doc) -> List[CausalRelation]:
        """Final contextual adjustments"""
        # This could include more sophisticated checks like:
        # - Checking for contradictory information
        # - Adjusting based on document topic/domain
        # - Using coreference resolution
        
        for relation in relations:
            # Simple adjustment: reduce confidence if cause and effect are very similar
            if self._strings_too_similar(relation.cause, relation.effect):
                relation.confidence *= 0.5
        
        return relations
    
    def _strings_too_similar(self, s1: str, s2: str) -> bool:
        """Check if two strings are too similar to be cause-effect"""
        s1_words = set(s1.lower().split())
        s2_words = set(s2.lower().split())
        
        if len(s1_words) == 0 or len(s2_words) == 0:
            return False
        
        overlap = len(s1_words & s2_words)
        total = len(s1_words | s2_words)
        
        similarity = overlap / total if total > 0 else 0
        return similarity > 0.8  # More than 80% word overlap
    
    def get_relation_summary(self, relations: List[CausalRelation]) -> Dict:
        """Get summary statistics of extracted relations"""
        if not relations:
            return {"total": 0, "by_type": {}, "avg_confidence": 0}
        
        by_type = defaultdict(int)
        total_confidence = 0
        
        for relation in relations:
            by_type[relation.relation_type] += 1
            total_confidence += relation.confidence
        
        return {
            "total": len(relations),
            "by_type": dict(by_type),
            "avg_confidence": total_confidence / len(relations),
            "high_confidence": len([r for r in relations if r.confidence > 0.7])
        }
