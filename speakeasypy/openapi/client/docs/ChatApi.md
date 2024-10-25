# speakeasypy.openapi.client.ChatApi

All URIs are relative to *http://localhost*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_api_room_with_roomid_with_since**](ChatApi.md#get_api_room_with_roomid_with_since) | **GET** /api/room/{roomId}/{since} | Get state and all messages for a chat room since a specified time
[**get_api_rooms**](ChatApi.md#get_api_rooms) | **GET** /api/rooms | Lists all Chatrooms for current user
[**post_api_room_with_roomid**](ChatApi.md#post_api_room_with_roomid) | **POST** /api/room/{roomId} | Post a message to a Chatroom.
[**post_api_room_with_roomid_reaction**](ChatApi.md#post_api_room_with_roomid_reaction) | **POST** /api/room/{roomId}/reaction | Post a chat message reaction to a Chatroom.


# **get_api_room_with_roomid_with_since**
> ChatRoomState get_api_room_with_roomid_with_since(room_id, since)

Get state and all messages for a chat room since a specified time

### Example

```python
import time
import speakeasypy.openapi.client
from speakeasypy.openapi.client.api import chat_api
from speakeasypy.openapi.client.model.error_status import ErrorStatus
from speakeasypy.openapi.client.model.chat_room_state import ChatRoomState
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = speakeasypy.openapi.client.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
with speakeasypy.openapi.client.ApiClient() as api_client:
    # Create an instance of the API class
    api_instance = chat_api.ChatApi(api_client)
    room_id = "roomId_example" # str | Id of the Chatroom
    since = 1 # int | Timestamp for new messages
    session = "session_example" # str | Session Token (optional)

    # example passing only required values which don't have defaults set
    try:
        # Get state and all messages for a chat room since a specified time
        api_response = api_instance.get_api_room_with_roomid_with_since(room_id, since)
        pprint(api_response)
    except speakeasypy.openapi.client.ApiException as e:
        print("Exception when calling ChatApi->get_api_room_with_roomid_with_since: %s\n" % e)

    # example passing only required values which don't have defaults set
    # and optional values
    try:
        # Get state and all messages for a chat room since a specified time
        api_response = api_instance.get_api_room_with_roomid_with_since(room_id, since, session=session)
        pprint(api_response)
    except speakeasypy.openapi.client.ApiException as e:
        print("Exception when calling ChatApi->get_api_room_with_roomid_with_since: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **room_id** | **str**| Id of the Chatroom |
 **since** | **int**| Timestamp for new messages |
 **session** | **str**| Session Token | [optional]

### Return type

[**ChatRoomState**](ChatRoomState.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json


### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | OK |  -  |
**401** | Unauthorized |  -  |
**404** | Not Found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_api_rooms**
> ChatRoomList get_api_rooms()

Lists all Chatrooms for current user

### Example

```python
import time
import speakeasypy.openapi.client
from speakeasypy.openapi.client.api import chat_api
from speakeasypy.openapi.client.model.error_status import ErrorStatus
from speakeasypy.openapi.client.model.chat_room_list import ChatRoomList
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = speakeasypy.openapi.client.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
with speakeasypy.openapi.client.ApiClient() as api_client:
    # Create an instance of the API class
    api_instance = chat_api.ChatApi(api_client)
    session = "session_example" # str | Session Token (optional)

    # example passing only required values which don't have defaults set
    # and optional values
    try:
        # Lists all Chatrooms for current user
        api_response = api_instance.get_api_rooms(session=session)
        pprint(api_response)
    except speakeasypy.openapi.client.ApiException as e:
        print("Exception when calling ChatApi->get_api_rooms: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **session** | **str**| Session Token | [optional]

### Return type

[**ChatRoomList**](ChatRoomList.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json


### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | OK |  -  |
**401** | Unauthorized |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **post_api_room_with_roomid**
> SuccessStatus post_api_room_with_roomid(room_id)

Post a message to a Chatroom.

### Example

```python
import time
import speakeasypy.openapi.client
from speakeasypy.openapi.client.api import chat_api
from speakeasypy.openapi.client.model.error_status import ErrorStatus
from speakeasypy.openapi.client.model.success_status import SuccessStatus
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = speakeasypy.openapi.client.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
with speakeasypy.openapi.client.ApiClient() as api_client:
    # Create an instance of the API class
    api_instance = chat_api.ChatApi(api_client)
    room_id = "roomId_example" # str | Id of the Chatroom
    session = "session_example" # str | Session Token (optional)
    body = "body_example" # str |  (optional)

    # example passing only required values which don't have defaults set
    try:
        # Post a message to a Chatroom.
        api_response = api_instance.post_api_room_with_roomid(room_id)
        pprint(api_response)
    except speakeasypy.openapi.client.ApiException as e:
        print("Exception when calling ChatApi->post_api_room_with_roomid: %s\n" % e)

    # example passing only required values which don't have defaults set
    # and optional values
    try:
        # Post a message to a Chatroom.
        api_response = api_instance.post_api_room_with_roomid(room_id, session=session, body=body)
        pprint(api_response)
    except speakeasypy.openapi.client.ApiException as e:
        print("Exception when calling ChatApi->post_api_room_with_roomid: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **room_id** | **str**| Id of the Chatroom |
 **session** | **str**| Session Token | [optional]
 **body** | **str**|  | [optional]

### Return type

[**SuccessStatus**](SuccessStatus.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: text/plain
 - **Accept**: application/json


### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | OK |  -  |
**401** | Unauthorized |  -  |
**404** | Not Found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **post_api_room_with_roomid_reaction**
> SuccessStatus post_api_room_with_roomid_reaction(room_id)

Post a chat message reaction to a Chatroom.

### Example

```python
import time
import speakeasypy.openapi.client
from speakeasypy.openapi.client.api import chat_api
from speakeasypy.openapi.client.model.error_status import ErrorStatus
from speakeasypy.openapi.client.model.success_status import SuccessStatus
from speakeasypy.openapi.client.model.chat_message_reaction import ChatMessageReaction
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = speakeasypy.openapi.client.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
with speakeasypy.openapi.client.ApiClient() as api_client:
    # Create an instance of the API class
    api_instance = chat_api.ChatApi(api_client)
    room_id = "roomId_example" # str | Id of the Chatroom
    session = "session_example" # str | Session Token (optional)
    chat_message_reaction = ChatMessageReaction(
        message_ordinal=1,
        type="THUMBS_UP",
    ) # ChatMessageReaction |  (optional)

    # example passing only required values which don't have defaults set
    try:
        # Post a chat message reaction to a Chatroom.
        api_response = api_instance.post_api_room_with_roomid_reaction(room_id)
        pprint(api_response)
    except speakeasypy.openapi.client.ApiException as e:
        print("Exception when calling ChatApi->post_api_room_with_roomid_reaction: %s\n" % e)

    # example passing only required values which don't have defaults set
    # and optional values
    try:
        # Post a chat message reaction to a Chatroom.
        api_response = api_instance.post_api_room_with_roomid_reaction(room_id, session=session, chat_message_reaction=chat_message_reaction)
        pprint(api_response)
    except speakeasypy.openapi.client.ApiException as e:
        print("Exception when calling ChatApi->post_api_room_with_roomid_reaction: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **room_id** | **str**| Id of the Chatroom |
 **session** | **str**| Session Token | [optional]
 **chat_message_reaction** | [**ChatMessageReaction**](ChatMessageReaction.md)|  | [optional]

### Return type

[**SuccessStatus**](SuccessStatus.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json


### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | OK |  -  |
**401** | Unauthorized |  -  |
**404** | Not Found |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

