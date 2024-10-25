# Speakeasy Python Client Library

## Getting started
### 1. Install

We provide a pre-built `whl` file for installation since `speakeasypy` has not been publicly released on PyPI for now. 
You can find this file at the following path: `speakeasy-python-client-library/dist/speakeasypy-1.0.0-py3-none-any.whl`

To install speakeasy in your local environment, use the following command:
```shell
pip install [local]/[path]/[to]/[your]/speakeasy-python-client-library/dist/speakeasypy-1.0.0-py3-none-any.whl
```
Please replace `[local]/[path]/[to]/[your]` with the actual path to the directory containing the `speakeasypy-1.0.0-py3-none-any.whl` file.

### 2. Initialize the Speakeasy Python framework and login

Please ensure that you are using the valid username and password of your bot.
```python
from speakeasypy import Speakeasy
speakeasy = Speakeasy(host='https://speakeasy.ifi.uzh.ch', username='name', password='pass')
speakeasy.login()  
# This framework will help you log out automatically when the program terminates.
```

### 3. Get chatrooms
```python
# Only check active chatrooms (i.e., remaining_time > 0) if active=True.
rooms = speakeasy.get_rooms(active=True)
```

### 4. Check messages and reactions in each chatroom, then post your messages to the corresponding room

```python
for room in rooms:
    # Retrieve messages from this chat room.
    # If only_partner=True, it filters out messages sent by the current bot.
    # If only_new=True, it filters out messages that have already been marked as processed.
    for message in room.get_messages(only_partner=True, only_new=True):
        # Implement your agent here #
        # Send a message to the corresponding chat room using the post_messages method of the room object.
        room.post_messages(f"Received your message: '{message.message}' ")
        # Mark the message as processed, so it will be filtered out when retrieving new messages.
        room.mark_as_processed(message)
    # Retrieve reactions from this chat room.
    # If only_new=True, it filters out reactions that have already been marked as processed.
    for reaction in room.get_reactions(only_new=True):
        # Implement your agent here #
        room.post_messages(f"Received your reaction: '{reaction.type}' ")
        room.mark_as_processed(reaction)
```

*Note: Each API endpoint has an embedded rate limit. If the rate of calls to an endpoint (e.g., `get_rooms()`) 
exceeds this limit, the returned result will be replaced with a cached value.

### 5. Additional Use Case
You can find a more comprehensive use case in `speakeasy-python-client-library/usecases/demo_bot.py`.

## Documentation for Relevant Classes

### Class Speakeasy
The `Speakeasy` class is the main entry point for `speakeasypy` library.

#### Methods
| Method      | Description                           | Parameters                                                                                                           | Returns                                                                   |
|-------------|---------------------------------------|----------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| `login`     | Logs in to the Speakeasy platform.    | None                                                                                                                 | `str`: Session token.                                                     |
| `logout`    | Logs out from the Speakeasy platform. | None                                                                                                                 | None                                                                      |
| `get_rooms` | Retrieves a list of chat rooms.       | `active` (bool, optional): If `True`, returns active chat rooms (rooms with remaining time > 0). Defaults to `True`. | `List[Chatroom]`: A list of Chatroom objects representing the chat rooms. |


### Class Chatroom

#### Methods
| Method              | Description                                          | Parameters                                                                                                                                                                                                            | Returns                                                        |
|---------------------|------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| `get_messages`      | Retrieves chat messages from the chatroom.           | `only_partner` (bool, optional): If `True`, returns messages from the chat partner only. Defaults to `True`. <br> `only_new` (bool, optional): If `True`, returns only new, unprocessed messages. Defaults to `True`. | `List[RestChatMessage]`: A list of chat messages.              |
| `get_reactions`     | Retrieves reactions from the chatroom.               | `only_new` (bool, optional): If `True`, returns only new, unprocessed reactions. Defaults to `True`.                                                                                                                  | `List[ChatMessageReaction]`: A list of chat message reactions. |
| `post_messages`     | Posts a message to the chatroom.                     | `message` (str): The message to be posted.                                                                                                                                                                            | None                                                           |
| `mark_as_processed` | Marks a message or reaction as processed.            | `msg_or_rec` (RestChatMessage or ChatMessageReaction]): The message or reaction to mark as processed.                                                                                                                 | None                                                           |
| `get_chat_partner`  | Gets the alias of your chat partner in the chatroom. | None                                                                                                                                                                                                                  | `str`: The alias of your chat partner.                         |

#### Properties
| Property Name    | Description                                                                                             | Type        |
|------------------|---------------------------------------------------------------------------------------------------------|-------------|
| `room_id`        | A unique identifier for the chatroom.                                                                   | `str`       |
| `my_alias`       | The alias of this bot for the chatroom.                                                                 | `str`       |
| `prompt`         | The prompt associated with the chatroom.                                                                | `str`       |
| `start_time`     | The starting time of the chatroom.                                                                      | `int`       |
| `remaining_time` | The remaining time for the chatroom's activity.                                                         | `int`       |
| `user_aliases`   | A list of user aliases participating in the chatroom (generally including a chat partner and your bot). | `List[str]` |
| `initiated`      | A flag indicating whether a welcome message has been sent.                                              | `bool`      |
| `session_token`  | The session token associated with the chatroom.                                                         | `str`       |

### Class RestChatMessage
#### Properties
| Property Name  | Type  |
|----------------|-------|
| `time_stamp`   | `int` |
| `author_alias` | `str` |
| `ordinal`      | `int` |
| `message`      | `str` |

### Class ChatMessageReaction
#### Properties
| Property Name     | Type                                                        |
|-------------------|-------------------------------------------------------------|
| `message_ordinal` | `int`                                                       |
| `type`            | `str` (possible values: "THUMBS_UP", "THUMBS_DOWN", "STAR") |

## Development for this package
This pacakge `speakeasypy` depends on an internal package `speakeasypy.openapi.client` which is generated by openapi. 
Therefore, developers need to re-build this internal package if the openapi specification (inputSpec) changed.

Install `speakeasypy` locally:

The following command generates `egg` files instead of `whl`.
```shell
python setup.py install
````

Distribute `speakeasypy` and test it locally:

The following command generates the source code and a `whl` file, then you can test it.
```shell
python setup.py sdist bdist_wheel
pip install [local]/[path]/[to]/[your]/dist/speakeasypy_xxx.whl
```

Note: make sure you have installed `wheel` for development.
```shell
pip install wheel
```

