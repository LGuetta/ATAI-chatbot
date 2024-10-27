import numpy as np
import csv
import rdflib
import spacy
import re
from fuzzywuzzy import process

class EmbeddingHandler:
    def __init__(self):
        # Load entity and relation embeddings
        try:
            self.entity_embeds = np.load("Datasets/ddis-graph-embeddings/entity_embeds.npy")
            self.relation_embeds = np.load("Datasets/ddis-graph-embeddings/relation_embeds.npy")
        except FileNotFoundError as e:
            print(f"Error loading embeddings: {str(e)}")
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
                        print(f"Conversion error on line: '{row}' with error: {str(e)}")
                        skipped_lines += 1
        if skipped_lines > 0:
            print(f"Skipped problematic lines during loading: {skipped_lines}")
        return mapping

    def load_entity_labels(self, file_path):
        graph = rdflib.Graph()
        try:
            graph.parse(file_path, format="turtle")
        except Exception as e:
            print(f"Error loading Turtle file: {str(e)}")
            return {}
        ent2lbl = {str(ent): str(lbl) for ent, lbl in graph.subject_objects(rdflib.RDFS.label)}
        return ent2lbl

    def get_entity_vector(self, entity_name):
        # Use fuzzy matching to find the closest entity
        best_match, score = process.extractOne(entity_name, self.lbl2ent.keys())
        if score > 80:  # Consider match if score is high enough
            uri = self.lbl2ent[best_match]
            if uri in self.entity_ids:
                index = self.entity_ids[uri]
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

# Test loading and entity extraction
if __name__ == "__main__":
    handler = EmbeddingHandler()
    if handler.entity_embeds is not None and handler.relation_embeds is not None:
        print("Embeddings loaded successfully.")

    # Test entity extraction
    query = "Who is the director of Inception?"
    entities = handler.extract_entities(query)
    print(f"Extracted entities: {entities}")
