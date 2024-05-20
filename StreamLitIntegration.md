Trainer provided requirements for associates to code on their own designed to reinforce practical application of technology.

# Langchain 006: StreamLit Integration Excercise

### Contributors:

William Terry

{ Reviewers }

### Prerequisites

1. Python
2. ChatGPT
3. Prompt Engineering

### Technologies Used

1. Python
2. LangChain
3. GPT-3.5-turbo


### Time
2 hours

### Applied Skills

1. use StreamLit as an interface with LLMs
2. create an easily modifiable chat-gpt like clone that can be easily modified


### Steps

1. Input Open AI key as environmental variable 'OPENAI_API_KEY'
2. 
```command
pip install streamlit
```
3. Add default model to st.session_state and set our OpenAI API key from Streamlit secrets which is ./streamit/secrets.toml and only contains the line 
OPENAI_API_KEY = "{key here}"

```python
import streamlit as st
from openai import OpenAI

st.title("ChatGPT-like clone")

# Set OpenAI API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
```
4. Using streaming with predetermined responses from OpenAI, we stream the responses to the frontend. In the API call, we pass the model name we hardcoded in session state and pass the chat history as a list of messages. We also pass the role and content of each message in the chat history. Finally, OpenAI returns a stream of responses (split into chunks of tokens), which we iterate through and display.
```python
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            stream=True,
        ):
            if isinstance(response.choices[0].delta.model_dump(mode="json")['content'],str):
                full_response += response.choices[0].delta.model_dump(mode="json")['content']
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

```
5. open a terminal and in the directory of your python module  (name below is "StreamLitFun.py") type:
```command
streamlit run StreamLitFun.py
```


### Requirements
1. Create a streamlit session
2. concatenate messages in a stream and add to the session state messages
3. Launch this code in the browser to create a Chat GPT like interface
4. Use this app to integrate with your previous Langchain code from this unit

### Discussion & FAQ

[Streamlit docs](https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps)

### Solution
Solution code: below or StreamLitFun.py

```python
#https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps

from openai import OpenAI
import streamlit as st

st.title("ChatGPT-like clone")

client = OpenAI(
  api_key=st.secrets["OPENAI_API_KEY"],  # this is also the default, it can be omitted
)

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            stream=True,
        ):
            if isinstance(response.choices[0].delta.model_dump(mode="json")['content'],str):
                full_response += response.choices[0].delta.model_dump(mode="json")['content']
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
```