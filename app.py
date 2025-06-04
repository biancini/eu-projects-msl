"""Generates a streamlit web interface to chat with the bot, with logs shown on the right."""

import streamlit as st

from query import query_project

st.set_page_config(layout="wide")
st.title("EU Projects Answering Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your question..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response = query_project(prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
