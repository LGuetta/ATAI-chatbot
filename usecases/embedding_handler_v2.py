import numpy as np
import csv
import rdflib
import spacy
import re
from rapidfuzz import process
from sklearn.metrics.pairwise import cosine_similarity
import logging

class EmbeddingHandler:
    def __init__(self):
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        # Load entity and relation embeddings
        try:
            self.entity_embeds = np.load("Datasets/ddis-graph-embeddings/entity_embeds.npy")
            self.relation_embeds = np.load("Datasets/ddis-graph-embeddings/relation_embeds.npy")
        except FileNotFoundError as e:
            logging.error(f"Error loading embeddings: {str(e)}")
            self.entity_embeds = None
            self.relation_embeds = None
            return

        # Load entity and relation ID mappings
        self.entity_ids = self.load_mapping("Datasets/ddis-graph-embeddings/entity_ids.del")
        self.relation_ids = self.load_mapping("Datasets/ddis-graph-embeddings/relation_ids.del")

        # Load entity labels
        self.ent2lbl = self.load_entity_labels("Datasets/14_graph.ttl")
        self.lbl2ent = {lbl: ent for ent, lbl in self.ent2lbl.items()}

        # Initialize spaCy for entity extraction
        self.nlp = spacy.load("en_core_web_sm")


    def load_mapping(self, file_path):
        mapping = {}
        skipped_lines = 0
        with open(file_path, "r") as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) == 2:
                    index, uri = row
                    try:
                        mapping[uri] = int(index)
                    except ValueError as e:
                        logging.error(f"Conversion error on line: '{row}' with error: {str(e)}")
                        skipped_lines += 1
        if skipped_lines > 0:
            logging.info(f"Skipped problematic lines during loading: {skipped_lines}")
        return mapping

    def load_entity_labels(self, file_path):
        graph = rdflib.Graph()
        try:
            graph.parse(file_path, format="turtle")
        except Exception as e:
            logging.error(f"Error loading Turtle file: {str(e)}")
            return {}
        ent2lbl = {str(ent): str(lbl) for ent, lbl in graph.subject_objects(rdflib.RDFS.label)}
        return ent2lbl

    def get_entity_vector(self, entity_name):
        """
        Get the embedding vector for a given entity name.
        """
        # First, try exact matching
        entity_uri = self.lbl2ent.get(entity_name)
        if entity_uri and entity_uri in self.entity_ids:
            index = self.entity_ids[entity_uri]
            return self.entity_embeds[index]
        
        # If exact match not found, limit the labels considered in fuzzy matching
        # Limit to labels that start with the same first letter
        first_letter = entity_name[0].lower()
        candidate_labels = [label for label in self.lbl2ent.keys() if label.lower().startswith(first_letter)]
        
        # If no candidates found, consider all labels
        if not candidate_labels:
            candidate_labels = list(self.lbl2ent.keys())
        
        # Now perform fuzzy matching on the limited set
        match = process.extractOne(entity_name, candidate_labels)
        if match:
            best_match = match[0]
            score = match[1]
            if score >= 80:
                entity_uri = self.lbl2ent[best_match]
                if entity_uri in self.entity_ids:
                    index = self.entity_ids[entity_uri]
                    return self.entity_embeds[index]
        return None

    def get_relation_vector(self, relation_name):
        if relation_name in self.relation_ids:
            index = self.relation_ids[relation_name]
            return self.relation_embeds[index]
        else:
            return None

    def extract_entities(self, query):
        # Use spaCy to parse the query and extract named entities
        doc = self.nlp(query)
        entities = [ent.text for ent in doc.ents if ent.label_ in ("PERSON", "WORK_OF_ART", "ORG")]
        if not entities:
            # Try to manually extract titles in quotes if spaCy finds nothing
            match = re.search(r'"([^"]+)"', query)
            if match:
                entities.append(match.group(1))
        return entities

    def get_top_similar_entities(self, label, top_n=5):
        """
        Get the top N most similar entities to the given label.
        """
        vector = self.get_entity_vector(label)
        if vector is None:
            return None

        # Compute cosine similarities efficiently
        # Normalize the vectors
        norm_vectors = self.entity_embeds / np.linalg.norm(self.entity_embeds, axis=1, keepdims=True)
        vector_norm = vector / np.linalg.norm(vector)

        # Compute cosine similarities
        similarities = norm_vectors @ vector_norm

        # Get top N indices excluding the entity itself
        top_indices = similarities.argsort()[-(top_n + 1):][::-1]
        results = []
        for idx in top_indices:
            if np.array_equal(self.entity_embeds[idx], vector):
                continue  # Skip the entity itself
            entity_uri = next((uri for uri, index in self.entity_ids.items() if index == idx), None)
            if entity_uri:
                entity_label = self.ent2lbl.get(entity_uri, "")
                similarity_score = similarities[idx]
                results.append((entity_label, similarity_score))
            if len(results) == top_n:
                break
        return results