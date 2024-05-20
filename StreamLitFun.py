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