import streamlit as st

st.set_page_config(page_title="Hey Chatbot", page_icon="🦙", layout="centered")

st.title("Hey Chatbot 💬🦙")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ask me a question!"}]

if prompt := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])