import spacy
from speakeasypy import Speakeasy, Chatroom
from typing import List
import time
from rdflib import Graph
from embedding_handler_v2 import EmbeddingHandler
import re
from rapidfuzz import process, fuzz  # Import 'fuzz' along with 'process'
import logging
import threading
from speakeasypy.openapi.client.exceptions import ApiException

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Load spaCy for NLP
nlp = spacy.load("en_core_web_trf")

DEFAULT_HOST_URL = 'https://speakeasy.ifi.uzh.ch'
listen_freq = 2

class Agent:
    def __init__(self, username, password, graph_file):
        self.username = username
        self.graph_file = graph_file
        self.knowledge_graph_loaded = False  # Initialize flag
        self.initialization_complete = False  # Existing flag
    


        # Initialize Speakeasy
        try:
            self.speakeasy = Speakeasy(
                host=DEFAULT_HOST_URL, username=username, password=password
            )
            self.speakeasy.login()
            logging.info("Speakeasy login successful.")
        except Exception as e:
            logging.error(f"Error during Speakeasy login: {str(e)}")
            exit(1)

        # Start a background thread to load the heavy components
        threading.Thread(target=self.load_components, daemon=True).start()

    def load_components(self):
        """
        Load the knowledge graph and embeddings in a background thread.
        """
        # Load the knowledge graph
        self.graph = Graph()
        try:
            self.graph.parse(self.graph_file, format="turtle")
            logging.info("Knowledge graph loaded successfully.")
            self.knowledge_graph_loaded = True  # Set the flag here
        except Exception as e:
            logging.error(f"Error parsing the graph: {str(e)}")
            exit(1)

        # Initialize EmbeddingHandler
        self.embedding_handler = EmbeddingHandler()

        # Set initialization complete flag
        self.initialization_complete = True
        logging.info("Initialization complete.")




    def initial_listen(self):
        """
        Listen for new chatrooms and send an initialization message.
        """
        while not hasattr(self, 'initialization_complete'):
            rooms: List[Chatroom] = self.speakeasy.get_rooms(active=True)
            for room in rooms:
                if not hasattr(room, 'loading_message_sent'):
                    room.post_messages(
                        "Please wait, I'm loading. "
                        "I'll be ready to chat in a minute!"
                    )
                    room.loading_message_sent = True
            time.sleep(listen_freq)
        # Once initialization is complete, start the main listener
        self.listen()

    def listen(self):
        while True:
            rooms: List[Chatroom] = self.speakeasy.get_rooms(active=True)
            for room in rooms:
                # Initialize room attributes if they don't exist
                if not hasattr(room, 'initiated'):
                    room.initiated = False
                if not hasattr(room, 'loading_message_sent'):
                    room.loading_message_sent = False
                if not hasattr(room, 'graph_loaded_message_sent'):
                    room.graph_loaded_message_sent = False

                # Send loading message if not already sent
                if not self.knowledge_graph_loaded and not room.loading_message_sent:
                    try:
                        room.post_messages(
                            "Please wait, I'm loading. "
                            "I'll be ready to chat shortly!"
                        )
                        room.loading_message_sent = True
                    except Exception as e:
                        logging.error(f"Error posting loading message to room {room.room_id}: {e}")

                # Send knowledge graph loaded message as soon as it's loaded
                if self.knowledge_graph_loaded and not room.graph_loaded_message_sent:
                    try:
                        room.post_messages("Knowledge graph loaded successfully :D")
                        room.graph_loaded_message_sent = True
                    except Exception as e:
                        logging.error(f"Error posting graph loaded message to room {room.room_id}: {e}")

                # Send welcome message after full initialization
                if self.initialization_complete and not room.initiated:
                    try:
                        room.post_messages(
                            f'Hello! This is an incredible welcome message from {room.my_alias}.'
                        )
                        room.initiated = True
                    except Exception as e:
                        logging.error(f"Error posting welcome message to room {room.room_id}: {e}")

                # Only process messages if initialization is complete and the room is initiated
                if self.initialization_complete and room.initiated:
                    for message in room.get_messages(only_partner=True, only_new=True):
                        print(
                            f"\t- Chatroom {room.room_id} "
                            f"- new message #{message.ordinal}: '{message.message}' "
                            f"- {self.get_time()}"
                        )

                        response = self.handle_query(message.message)
                        if response is None:
                            response = "I'm sorry, I couldn't understand the question. Could you please rephrase it?"
                        try:
                            room.post_messages(response)
                        except Exception as e:
                            logging.error(f"Error posting response to room {room.room_id}: {e}")
                        room.mark_as_processed(message)

                    for reaction in room.get_reactions(only_new=True):
                        print(
                            f"\t- Chatroom {room.room_id} "
                            f"- new reaction #{reaction.message_ordinal}: "
                            f"'{reaction.type}' - {self.get_time()}"
                        )

                        try:
                            room.post_messages(f"Received your reaction: '{reaction.type}'")
                        except Exception as e:
                            logging.error(f"Error posting reaction to room {room.room_id}: {e}")
                        room.mark_as_processed(reaction)

            time.sleep(listen_freq)


    def handle_query(self, query):
        entity = None  # Initialize entity

        # Analyze the question using NLP
        doc = nlp(query)

        # Define unwanted entities to exclude
        unwanted_entities = set([
            'who', 'what', 'when', 'where', 'why', 'how',
            'director', 'screenwriter', 'writer', 'author',
            'released', 'release date', 'published', 'movie', 'film', 'wrote', 'directed',
        ])

        # Prima cerca di estrarre il titolo tra virgolette
        match = re.search(r'"([^"]+)"', query)
        if match:
            entity = match.group(1)
            print(f"Entity extracted from quotes: {entity}")

        # Se non trova entità tra virgolette, usa spaCy
        if not entity:
            entities = [ent.text for ent in doc.ents if ent.label_ in ["WORK_OF_ART", "ORG", "EVENT"]]
            entities = [ent for ent in entities if ent.lower() not in unwanted_entities]
            print(f"Extracted entities after spaCy: {entities}")

            # If entities are found, proceed
            if entities:
                entity = entities[0].strip()
            else:
                # Collect proper nouns and adjacent tokens to form potential entities
                proper_nouns = []
                temp_entity = ''
                for token in doc:
                    # Include tokens that are proper nouns, numbers, adjectives, or punctuation that might be part of the title
                    if token.pos_ in ['PROPN', 'NOUN', 'NUM', 'ADJ'] or token.text in [':', '-', '"', "'"]:
                        temp_entity += token.text_with_ws
                    elif token.is_punct and token.text in [":", "-", ","]:
                        temp_entity += token.text_with_ws
                    else:
                        if temp_entity.strip():
                            proper_nouns.append(temp_entity.strip(' "\''))
                            temp_entity = ''
                if temp_entity.strip():
                    proper_nouns.append(temp_entity.strip(' "\''))

                # Unire i nomi propri in caso contengano parti divise dai due punti
                proper_nouns_combined = " ".join(proper_nouns)
                print(f"Proper nouns combined: {proper_nouns_combined}")

                # Use the combined proper nouns as the entity if they exist
                if proper_nouns_combined:
                    entity = proper_nouns_combined
                else:
                    # Use the entire query for fuzzy matching as a last resort
                    best_match = self.handle_fuzzy_matching(query)
                    if best_match:
                        entity = best_match
                    else:
                        return "Sorry, I couldn't find any entity in your question."

        entity = entity.strip()
        print(f"Final selected entity: {entity}")
        
        # Determine the type of request
        query_lower = query.lower()
        if "director" in query_lower:
            factual_answer = self.get_director(entity)
            embedding_answer = self.handle_embedding_query(entity)
            return f"{factual_answer}\n{embedding_answer}"

        elif any(keyword in query_lower for keyword in ["screenwriter", "writer", "author"]):
            factual_answer = self.get_screenwriter(entity)
            embedding_answer = self.handle_embedding_query(entity)
            return f"{factual_answer}\n{embedding_answer}"

        elif any(keyword in query_lower for keyword in ["released", "release date", "published"]):
            factual_answer = self.get_release_date(entity)
            embedding_answer = self.handle_embedding_query(entity)
            return f"{factual_answer}\n{embedding_answer}"

        else:
            # If it's a statement or an embedding question
            embedding_answer = self.handle_embedding_query(entity)
            return embedding_answer






    def get_director(self, entity_label):
        """
        Fetch the director of the specified film entity using label filtering.
        """
        # Escape special characters in entity_label
        entity_label_escaped = entity_label.replace('"', '\\"')

        sparql_query = f'''
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX ns1: <http://www.wikidata.org/prop/direct/>

        SELECT ?directorLabel WHERE {{
            ?film rdfs:label ?label .
            FILTER (lcase(str(?label)) = lcase("{entity_label_escaped}") && lang(?label) = "en") .

            ?film ns1:P57 ?director .
            ?director rdfs:label ?directorLabel .
            FILTER (lang(?directorLabel) = "en")
        }}
        '''
        factual_answer = self.execute_sparql_query(sparql_query)

        if factual_answer == "No results found.":
            # Provide alternative information
            description = self.get_description(entity_label)
            return f"Factual Answer: Sorry, I couldn't find the director information for '{entity_label}'.\n{description}"

        unique_answers = list(set(factual_answer.split(", ")))
        return f"Factual Answer: The director of '{entity_label}' is {', '.join(unique_answers)}."




    def get_screenwriter(self, entity_label):
        """
        Fetch the screenwriter of the specified film entity using label filtering.
        """
        # Escape special characters in entity_label
        entity_label_escaped = entity_label.replace('"', '\\"')

        sparql_query = f'''
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX ns1: <http://www.wikidata.org/prop/direct/>

        SELECT ?screenwriterLabel WHERE {{
            ?film rdfs:label ?label .
            FILTER (lcase(str(?label)) = lcase("{entity_label_escaped}") && lang(?label) = "en") .

            ?film ns1:P58 ?screenwriter .
            ?screenwriter rdfs:label ?screenwriterLabel .
            FILTER (lang(?screenwriterLabel) = "en")
        }}
        '''
        factual_answer = self.execute_sparql_query(sparql_query)

        if factual_answer == "No results found.":
            description = self.get_description(entity_label)
            return f"Factual Answer: Sorry, I couldn't find the screenwriter information for '{entity_label}'.\n{description}"

        unique_answers = list(set(factual_answer.split(", ")))
        return f"Factual Answer: The screenwriter of '{entity_label}' is {', '.join(unique_answers)}."


    def get_release_date(self, entity_label):
        """
        Fetch the release date of the specified film entity using label filtering.
        """
        # Escape special characters in entity_label
        entity_label_escaped = entity_label.replace('"', '\\"')

        sparql_query = f'''
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX ns1: <http://www.wikidata.org/prop/direct/>

        SELECT ?releaseDate WHERE {{
            ?film rdfs:label ?label .
            FILTER (lcase(str(?label)) = lcase("{entity_label_escaped}") && lang(?label) = "en") .

            ?film ns1:P577 ?releaseDate .
        }}
        '''
        factual_answer = self.execute_sparql_query(sparql_query)

        if factual_answer == "No results found.":
            description = self.get_description(entity_label)
            return f"Factual Answer: Sorry, I couldn't find the release date for '{entity_label}'.\n{description}"

        unique_answers = list(set(factual_answer.split(", ")))
        return f"Factual Answer: The release date of '{entity_label}' is {', '.join(unique_answers)}."


    def get_description(self, entity_label):
        """
        Fetch the description of the specified film entity.
        """
        # Escape special characters in entity_label
        entity_label_escaped = entity_label.replace('"', '\\"')

        sparql_query = f'''
        PREFIX ns2: <http://schema.org/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?description WHERE {{
            ?film rdfs:label ?label .
            FILTER (lcase(str(?label)) = lcase("{entity_label_escaped}") && lang(?label) = "en") .

            ?film ns2:description ?description .
            FILTER (lang(?description) = "en")
        }}
        '''
        description = self.execute_sparql_query(sparql_query)
        if description == "No results found.":
            return "No description available."
        else:
            return f"Description: {description}"


    def execute_sparql_query(self, sparql_query):
        """
        Execute the SPARQL query and return results from the knowledge graph.
        """
        try:
            logging.info(f"Executing SPARQL query: {sparql_query}")
            result = self.graph.query(sparql_query)
            result_list = [str(row[0]) for row in result]
            return ", ".join(result_list) if result_list else "No results found."
        except Exception as e:
            logging.error(f"Error executing SPARQL query: {e}")
            return "No results found."

    def handle_embedding_query(self, entity):
        """
        Find related information using entity embeddings.
        """
        similar_entities = self.embedding_handler.get_top_similar_entities(entity, top_n=5)
        if not similar_entities:
            return "(Embedding Answer) No similar entities found."
        else:
            response = f"(Embedding Answer) Entities similar to '{entity}':\n"
            for idx, (other_entity, score) in enumerate(similar_entities, 1):
                response += f"{idx}. {other_entity} (Similarity: {score:.2f})\n"
            return response

    def handle_fuzzy_matching(self, text):
        # Define unwanted entities to exclude
        unwanted_entities = set([
            'director', 'screenwriter', 'writer', 'author',
            'released', 'release date', 'published', 'movie', 'film',
        ])

        if not self.embedding_handler.lbl2ent:
            return None
        all_labels = list(self.embedding_handler.lbl2ent.keys())
        all_labels = [label for label in all_labels if label.lower() not in unwanted_entities]

        # Ignore text that's too short or empty
        if not text or len(text) < 3:
            return None

        match = process.extractOne(text, all_labels, scorer=fuzz.WRatio)
        if match:
            best_match = match[0]
            score = match[1]
            if score > 80:
                return best_match
        return None






    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S %d-%m-%Y", time.localtime())

if __name__ == '__main__':
    demo_bot = Agent("dark-star", "H9krY2I3", "Datasets/14_graph.ttl")
    demo_bot.listen()
