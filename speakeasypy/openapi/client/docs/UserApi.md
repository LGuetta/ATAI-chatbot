# speakeasypy.openapi.client.UserApi

All URIs are relative to *http://localhost*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_api_logout**](UserApi.md#get_api_logout) | **GET** /api/logout | Clears all user roles of the current session.
[**get_api_user_current**](UserApi.md#get_api_user_current) | **GET** /api/user/current | Returns details for the current session.
[**post_api_login**](UserApi.md#post_api_login) | **POST** /api/login | Sets roles for session based on user account and returns a session cookie.


# **get_api_logout**
> SuccessStatus get_api_logout()

Clears all user roles of the current session.

### Example

```python
import time
import speakeasypy.openapi.client
from speakeasypy.openapi.client.api import user_api
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
    api_instance = user_api.UserApi(api_client)
    session = "session_example" # str | Session Token (optional)

    # example passing only required values which don't have defaults set
    # and optional values
    try:
        # Clears all user roles of the current session.
        api_response = api_instance.get_api_logout(session=session)
        pprint(api_response)
    except speakeasypy.openapi.client.ApiException as e:
        print("Exception when calling UserApi->get_api_logout: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **session** | **str**| Session Token | [optional]

### Return type

[**SuccessStatus**](SuccessStatus.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json


### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | OK |  -  |
**400** | Bad Request |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_api_user_current**
> UserSessionDetails get_api_user_current()

Returns details for the current session.

### Example

```python
import time
import speakeasypy.openapi.client
from speakeasypy.openapi.client.api import user_api
from speakeasypy.openapi.client.model.error_status import ErrorStatus
from speakeasypy.openapi.client.model.user_session_details import UserSessionDetails
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = speakeasypy.openapi.client.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
with speakeasypy.openapi.client.ApiClient() as api_client:
    # Create an instance of the API class
    api_instance = user_api.UserApi(api_client)
    session = "session_example" # str | Session Token (optional)

    # example passing only required values which don't have defaults set
    # and optional values
    try:
        # Returns details for the current session.
        api_response = api_instance.get_api_user_current(session=session)
        pprint(api_response)
    except speakeasypy.openapi.client.ApiException as e:
        print("Exception when calling UserApi->get_api_user_current: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **session** | **str**| Session Token | [optional]

### Return type

[**UserSessionDetails**](UserSessionDetails.md)

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

# **post_api_login**
> UserSessionDetails post_api_login()

Sets roles for session based on user account and returns a session cookie.

### Example

```python
import time
import speakeasypy.openapi.client
from speakeasypy.openapi.client.api import user_api
from speakeasypy.openapi.client.model.error_status import ErrorStatus
from speakeasypy.openapi.client.model.user_session_details import UserSessionDetails
from speakeasypy.openapi.client.model.login_request import LoginRequest
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = speakeasypy.openapi.client.Configuration(
    host = "http://localhost"
)


# Enter a context with an instance of the API client
with speakeasypy.openapi.client.ApiClient() as api_client:
    # Create an instance of the API class
    api_instance = user_api.UserApi(api_client)
    login_request = LoginRequest(
        username="username_example",
        password="password_example",
    ) # LoginRequest |  (optional)

    # example passing only required values which don't have defaults set
    # and optional values
    try:
        # Sets roles for session based on user account and returns a session cookie.
        api_response = api_instance.post_api_login(login_request=login_request)
        pprint(api_response)
    except speakeasypy.openapi.client.ApiException as e:
        print("Exception when calling UserApi->post_api_login: %s\n" % e)
```


### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **login_request** | [**LoginRequest**](LoginRequest.md)|  | [optional]

### Return type

[**UserSessionDetails**](UserSessionDetails.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json


### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | OK |  -  |
**400** | Bad Request |  -  |
**401** | Unauthorized |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

