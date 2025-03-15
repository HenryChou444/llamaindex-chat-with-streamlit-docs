import glob

import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.readers.json import JSONReader

st.set_page_config(page_title="Chat with the Streamlit docs, powered by LlamaIndex", page_icon="🦙", layout="centered",
                   initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.openai_key
st.title("Hey Chatbot 💬")
# st.info("Check out the full tutorial to build this app in our [blog post](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/)", icon="📃")


if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Bonjour, je suis l'assistant Hey, posez-moi toutes les questions concernant,...!",
        }
    ]


@st.cache_resource(show_spinner=False)
def load_data():
    Settings.llm = OpenAI(
        model="gpt-3.5-turbo",
        temperature=0.2,
        system_prompt=
        """
        You are a chatbot for Hey-Telecom. Your role is to assist customers 24/7 by responding to
        the following requests: providing information about bills, accounts, troubleshooting, offers,
        and plans; recommending products, guiding purchases, and informing about promotions and discounts;
        activating new services and modifying plans; organizing in-store visits or installations and locating
        stores; collecting customer feedback, conducting surveys, and escalating complex requests; informing
        about service outages and network status; assisting with registration, SIM activation, and service
        activation; offering self-service account and service management options; providing tailored recommendations
        to meet each customer's needs and handling upselling opportunities; automating recurring requests. "
        If a request exceeds your capabilities, transfer it to a human agent.
        Keep your answers technical and based on facts – do not hallucinate features.
        """,)
    reader = JSONReader(
        levels_back=0,  # Set levels back as needed
        collapse_length=None,  # Set collapse length as needed
        ensure_ascii=False,  # ASCII encoding option
        is_jsonl=False,  # Set if input is JSON Lines format
        clean_json=True  # Clean up formatting-only lines
    )
    # Find all JSON files in the specified directory
    json_files = glob.glob(f"./data/testjson/*.json")
    # Load the data from each JSON file
    documents = []
    for json_file in json_files:
        documents.extend(reader.load_data(input_file=json_file, extra_info={}))

    # Create an index for querying
    index = VectorStoreIndex.from_documents(documents)
    return index


index = load_data()

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True, streaming=True
    )

if prompt := st.chat_input(
        "Ask a question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Write message history to UI
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        st.write_stream(response_stream.response_gen)
        message = {"role": "assistant", "content": response_stream.response}
        # Add response to message history
        st.session_state.messages.append(message)
