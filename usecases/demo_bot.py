import spacy
from speakeasypy import Speakeasy, Chatroom
from typing import List
import time
from rdflib import Graph
from embedding_handler_v2 import EmbeddingHandler
import re
from fuzzywuzzy import process

# Load spaCy for NLP
nlp = spacy.load("en_core_web_sm")

DEFAULT_HOST_URL = 'https://speakeasy.ifi.uzh.ch'
listen_freq = 2

class Agent:
    def __init__(self, username, password, graph_file):
        self.username = username
        # Initialize Speakeasy
        self.speakeasy = Speakeasy(host=DEFAULT_HOST_URL, username=username, password=password)
        self.speakeasy.login()

        # Load the knowledge graph
        self.graph = Graph()
        self.graph.parse(graph_file, format="turtle")

        # Initialize EmbeddingHandler
        self.embedding_handler = EmbeddingHandler()

    def listen(self):
        while True:
            rooms: List[Chatroom] = self.speakeasy.get_rooms(active=True)
            for room in rooms:
                if not room.initiated:
                    room.post_messages(f'Hello! This is a welcome message from {room.my_alias}.')
                    room.initiated = True

                for message in room.get_messages(only_partner=True, only_new=True):
                    print(
                        f"\t- Chatroom {room.room_id} "
                        f"- new message #{message.ordinal}: '{message.message}' "
                        f"- {self.get_time()}")
                    
                    response = self.handle_query(message.message)
                    room.post_messages(response)
                    room.mark_as_processed(message)

                for reaction in room.get_reactions(only_new=True):
                    print(
                        f"\t- Chatroom {room.room_id} "
                        f"- new reaction #{reaction.message_ordinal}: '{reaction.type}' "
                        f"- {self.get_time()}")

                    room.post_messages(f"Received your reaction: '{reaction.type}' ")
                    room.mark_as_processed(reaction)

            time.sleep(listen_freq)

    def handle_query(self, query):
        """
        Decide how to handle the query: SPARQL, Embeddings, or other.
        """
        # Analyze the question using NLP
        doc = nlp(query)
        
        # Try extracting named entities using spaCy
        entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "WORK_OF_ART", "ORG"]]
        
        # If no entities found by spaCy, use regex to find quoted titles
        if not entities:
            match = re.search(r'"([^"]+)"', query)
            if match:
                entities.append(match.group(1))

        # If still no entities found, apply fuzzy matching to correct spelling
        if not entities:
            all_labels = list(self.embedding_handler.lbl2ent.keys())
            best_match, score = process.extractOne(query, all_labels)
            if score > 70:  # Only consider it a match if it's a reasonably high score
                entities.append(best_match)

        if not entities:
            return "Sorry, I couldn't find any entity in your question."

        entity = entities[0]

        # Determine the type of request (e.g., director, screenwriter, release date)
        if "director" in query.lower():
            sparql_query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX dbo: <http://dbpedia.org/ontology/>
            SELECT ?director WHERE {{
                ?film rdfs:label "{entity}"@en .
                ?film dbo:director ?director .
            }}
            """
            factual_result = self.execute_sparql_query(sparql_query)
            return f"Factual Answer: {factual_result}"

        elif "released" in query.lower() or "release date" in query.lower():
            sparql_query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX dbo: <http://dbpedia.org/ontology/>
            SELECT ?releaseDate WHERE {{
                ?film rdfs:label "{entity}"@en .
                ?film dbo:releaseDate ?releaseDate .
            }}
            """
            factual_result = self.execute_sparql_query(sparql_query)
            return f"Factual Answer: {factual_result}"

        elif "screenwriter" in query.lower() or "writer" in query.lower():
            sparql_query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX dbo: <http://dbpedia.org/ontology/>
            SELECT ?writer WHERE {{
                ?film rdfs:label "{entity}"@en .
                ?film dbo:writer ?writer .
            }}
            """
            factual_result = self.execute_sparql_query(sparql_query)
            return f"Factual Answer: {factual_result}"

        # Attempt to answer with embeddings if SPARQL is not applicable
        embedding_result = self.handle_embedding_query(entity)
        if embedding_result:
            return embedding_result

        return "No suitable answer found."

    def execute_sparql_query(self, sparql_query):
        """
        Execute the SPARQL query and return results from the knowledge graph.
        """
        try:
            print(f"Executing SPARQL query: {sparql_query}")
            result = self.graph.query(sparql_query)
            result_list = [str(row[0]) for row in result]
            return ", ".join(result_list) if result_list else "No results found."
        except Exception as e:
            return f"Error executing query: {str(e)}"

    def handle_embedding_query(self, entity):
        """
        Find related information using entity embeddings.
        """
        entity_vector = self.embedding_handler.get_entity_vector(entity)
        if entity_vector is None:
            return f"Entity '{entity}' not found in the embeddings."

        similar_entities = [(other_entity, self.calculate_similarity(entity_vector, self.embedding_handler.get_entity_vector(other_entity)))
                            for other_entity in self.embedding_handler.entity_ids.keys() if self.embedding_handler.get_entity_vector(other_entity) is not None]

        similar_entities = sorted(similar_entities, key=lambda x: x[1], reverse=True)

        if similar_entities:
            most_similar_entity, similarity_score = similar_entities[0]
            return f"Based on embeddings, the most similar entity to '{entity}' is '{most_similar_entity}' with a similarity score of {similarity_score:.2f}. (Embedding Answer)"
        else:
            return "No similar entities found."

    def calculate_similarity(self, vector1, vector2):
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity([vector1], [vector2])[0][0]

    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())

if __name__ == '__main__':
    demo_bot = Agent("dark-star", "H9krY2I3", "Datasets/14_graph.ttl")
    demo_bot.listen()
