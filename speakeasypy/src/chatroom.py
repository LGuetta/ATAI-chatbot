import logging
import time

from datetime import datetime
from typing import List, Union
from speakeasypy.openapi.client.models import RestChatMessage, ChatMessageReaction


class Chatroom:
    def __init__(self,
                 room_id: str,
                 my_alias: str,
                 prompt: str,
                 start_time: int,
                 remaining_time: int,
                 user_aliases: List[str],
                 **kwargs
                 ):
        """Chatroom - a model representing a chatroom for bots to interact.

        Args:
            room_id (str): A unique identifier for the chatroom.
            my_alias (str): The alias of this bot for the chatroom.
            prompt (str): The prompt associated with the chatroom.
            start_time (int): The starting time of the chatroom.
            remaining_time (int): The remaining time for the chatroom's activity.
            user_aliases (List[str]): A list of user aliases participating in the chatroom (generally including a chat partner and your bot).
        """

        self.room_id = room_id
        self.my_alias = my_alias
        self.prompt = prompt
        self.start_time = start_time
        self.remaining_time = remaining_time
        self.user_aliases = user_aliases
        self.initiated = False  # This flag indicates whether a welcome message has been sent

        logging.basicConfig(level=logging.INFO)

        self.session_token = kwargs.get('session_token', None)
        self.chat_api = kwargs.get('chat_api', None)
        if self.session_token is None or self.chat_api is None:
            logging.error(f"No session_token or chat_api for chatroom {self.room_id}, "
                          f"api requests by this chatroom will result in an error")
        # Store ordinals for processed messages and reactions to exclude them from "new" messages.
        self.processed_ordinals = {
            'messages': [],
            'reactions': [],
        }

        self.__request_limit = kwargs.get('request_limit', 1)  # seconds
        self.__state_api_cache = None  # ChatRoomState (including messages and reactions from api call)
        self.__last_msg_timestamp = 0
        self.__last_state_call = 0
        self.__last_post_call = 0

    def __update_chat_room_state(self):
        """ Cache the state of this room and implement a request rate limit for this API call. """
        if not self.session_token:
            logging.error(f"This room {self.room_id} has no active session. Updating room state failed.")
            return
        current_time = time.time()
        elapsed_time = current_time - self.__last_state_call
        if elapsed_time < self.__request_limit and self.__state_api_cache is not None:
            return

        try:
            response = self.chat_api.get_api_room_with_roomid_with_since(
                room_id=self.room_id, since=self.__last_msg_timestamp, session=self.session_token)
            if response:
                if self.__state_api_cache is None:
                    self.__state_api_cache = response
                else:
                    # The reactions returned by the backend have nothing to do with the "since" parameter for now,
                    # so just copy all reactions here.
                    self.__state_api_cache.reactions = response.reactions
                    # Append new messages and update the last timestamp
                    for m in response.messages:
                        if m.ordinal not in [msg.ordinal for msg in self.__state_api_cache.messages]:
                            self.__state_api_cache.messages.append(m)
                            self.__last_msg_timestamp = max(self.__last_msg_timestamp, m.time_stamp)
            else:
                logging.error(f"Failed to update the state of room {self.room_id}.")
            self.__last_state_call = current_time
        except Exception as e:
            logging.error(f"An error occurred while updating the state of room {self.room_id}: {e}")

    def get_messages(self, only_partner=True, only_new=True) -> List[RestChatMessage]:
        self.__update_chat_room_state()
        if self.__state_api_cache is None:
            logging.error(f"Updating room state failed. No messages in room {self.room_id}.")
            return []

        filtered_messages = self.__state_api_cache.messages

        if only_partner:  # TODO: openAPI will automatically converts 'authorAlias' to 'author_alias'
            filtered_messages = [message for message in filtered_messages if message.author_alias != self.my_alias]

        if only_new:
            filtered_messages = [message for message in filtered_messages if
                                 message.ordinal not in self.processed_ordinals['messages']]

        return filtered_messages

    def get_reactions(self, only_new=True) -> List[ChatMessageReaction]:
        self.__update_chat_room_state()
        if self.__state_api_cache is None:
            logging.error(f"Updating room state failed. No reactions in room {self.room_id}.")
            return []

        filtered_reactions = self.__state_api_cache.reactions
        if only_new:
            filtered_reactions = [reaction for reaction in filtered_reactions if
                                  reaction.message_ordinal not in self.processed_ordinals['reactions']]
        return filtered_reactions

    def post_messages(self, message):
        if self.session_token:
            # Check if the time elapsed since the last post call is less than the request limit.
            current_time = time.time()
            elapsed_time = current_time - self.__last_post_call
            # If elapsed time is less than the request limit, sleep for the remaining time to enforce rate limiting.
            if elapsed_time < self.__request_limit:
                time.sleep(self.__request_limit - elapsed_time)
                print(f"(Sleep {self.__request_limit - elapsed_time} secs to avoid posting requests too frequently.)")
            try:
                response = self.chat_api.post_api_room_with_roomid(
                    room_id=self.room_id, session=self.session_token, body=message)
                if not response:
                    logging.error(f"Failed to post message to room {self.room_id}.")
            except Exception as e:
                logging.error(f"An error occurred while posting the message to room {self.room_id}:", e)

            self.__last_post_call = time.time()  # store the completed time
        else:
            logging.error(f"This room {self.room_id} has no active session. Posting messages failed.")

    def mark_as_processed(self, msg_or_rec: Union[RestChatMessage, ChatMessageReaction]):
        if isinstance(msg_or_rec, RestChatMessage):
            self.processed_ordinals['messages'].append(msg_or_rec.ordinal)
        elif isinstance(msg_or_rec, ChatMessageReaction):
            self.processed_ordinals['reactions'].append(msg_or_rec.message_ordinal)
        else:
            logging.error("Please pass a message or reaction object to mark it as processed.")

    def get_chat_partner(self) -> str:
        # get the alias of your chat partner
        return next(alias for alias in self.user_aliases if alias != self.my_alias)

    def __eq__(self, other):
        if isinstance(other, Chatroom):
            return self.room_id == other.room_id
        return False

    def __contains__(self, chatroom_list):
        return any(self == room for room in chatroom_list)

    def __str__(self):
        start_time_formatted = datetime.fromtimestamp(self.start_time // 1000).strftime("%H:%M:%S, %d-%m-%Y")
        remaining_min, remaining_sec = divmod(self.remaining_time // 1000, 60)

        return f"""
        room_id: {self.room_id};
        my_alias: {self.my_alias};
        prompt: {self.prompt};
        start_time: {start_time_formatted};
        remaining_time: {remaining_min}min {remaining_sec}sec.
        """

    def __repr__(self):
        return str(self)
