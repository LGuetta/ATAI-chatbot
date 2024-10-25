from speakeasypy import Speakeasy, Chatroom
from typing import List
import time
from rdflib import Graph

DEFAULT_HOST_URL = 'https://speakeasy.ifi.uzh.ch'
listen_freq = 2


class Agent:
    def __init__(self, username, password, graph_file):
        self.username = username
        # Initialize the Speakeasy Python framework and login.
        self.speakeasy = Speakeasy(host=DEFAULT_HOST_URL, username=username, password=password)
        self.speakeasy.login()  # This framework will help you log out automatically when the program terminates.

        # Load the knowledge graph
        self.graph = Graph()
        self.graph.parse(graph_file, format="turtle")
        # self.graph.serialize("14_graph_converted.nt", format="nt")

    def listen(self):
        while True:
            # only check active chatrooms (i.e., remaining_time > 0) if active=True.
            rooms: List[Chatroom] = self.speakeasy.get_rooms(active=True)
            for room in rooms:
                if not room.initiated:
                    # send a welcome message if room is not initiated
                    room.post_messages(f'Hello! This is a welcome message from {room.my_alias}.')
                    room.initiated = True
                # Retrieve messages from this chat room.
                # If only_partner=True, it filters out messages sent by the current bot.
                # If only_new=True, it filters out messages that have already been marked as processed.
                for message in room.get_messages(only_partner=True, only_new=True):
                    print(
                        f"\t- Chatroom {room.room_id} "
                        f"- new message #{message.ordinal}: '{message.message}' "
                        f"- {self.get_time()}")

                    # Implement your agent here #

                    # Send a message to the corresponding chat room using the post_messages method of the room object.
                    # room.post_messages(f"Received your message: '{message.message}' ")
                    
                    # Process SPARQL query
                    query_result = self.handle_sparql_query(message.message)
                    if query_result:
                        room.post_messages(f"{query_result}")
                    else:
                        room.post_messages(f"Invalid query or no results for: '{message.message}'")
                    
                    # Mark the message as processed, so it will be filtered out when retrieving new messages.
                    room.mark_as_processed(message)

                # Retrieve reactions from this chat room.
                # If only_new=True, it filters out reactions that have already been marked as processed.
                for reaction in room.get_reactions(only_new=True):
                    print(
                        f"\t- Chatroom {room.room_id} "
                        f"- new reaction #{reaction.message_ordinal}: '{reaction.type}' "
                        f"- {self.get_time()}")

                    # Implement your agent here #

                    room.post_messages(f"Received your reaction: '{reaction.type}' ")
                    room.mark_as_processed(reaction)

            time.sleep(listen_freq)
    
    def handle_sparql_query(self, query):
        """
        Process the SPARQL query and return results from the knowledge graph.
        """
        try:
            print(f"Executing SPARQL query: {query}")

            # Execute SPARQL query
            result = self.graph.query(query)
            # Format the results
            result_list = []
            for row in result:
                result_list.append(", ".join(str(item) for item in row))
            return "\n".join(result_list) if result_list else "No results found."
        except Exception as e:
            return f"Error executing query: {str(e)}"

    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())


if __name__ == '__main__':
    demo_bot = Agent("dark-star", "H9krY2I3","14_graph.ttl")
    demo_bot.listen()
