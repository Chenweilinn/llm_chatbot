import streamlit as st
from llm_chains import load_normal_chain, load_pdf_chat_chain
from langchain.memory import StreamlitChatMessageHistory
import yaml
import os
from utils import save_chat_history_json, get_timestamps , load_chat_history_json
from pdf_handler import add_documents_to_db

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def load_chain(chat_history):
    if st.session_state.pdf_chat:
        return load_pdf_chat_chain(chat_history)
    return load_normal_chain(chat_history)


def clear_input_fileld():
    st.session_state.user_question = st.session_state.user_input
    st.session_state.user_input = ""

def set_send_input():
    st.session_state.send_input =  True
    clear_input_fileld()

def toggle_pdf_chat():
    st.session_state.pdf_chat = True

def save_chat_history():
    if st.session_state.history != []:
        if st.session_state.session_key == "new_session":
            st.session_state.new_session_key = get_timestamps()+ ".json"
            save_chat_history_json(chat_history=st.session_state.history, file_path=config["chat_history_path"] + st.session_state.new_session_key)
        
        else:
            save_chat_history_json(chat_history=st.session_state.history, file_path=config["chat_history_path"] +  st.session_state.session_key)

def track_index():
    st.session_state.session_index_tracker = st.session_state.session_key

def main():
    st.title("ADS LLM ChatBot")
    chat_container = st.container()
    #add chat sessions in sidebar
    st.sidebar.title("Chat Sessions")
    chat_sessions =["new_session"] + os.listdir(config["chat_history_path"])

    if "send_input" not in st.session_state:
        st.session_state.session_key = "new_session"
        st.session_state.send_input = False
        st.session_state.user_question = ""
        st.session_state.new_session_key = None
        st.session_state.session_index_tracker = "new_session" 
    if st.session_state.session_key == "new_session" and st.session_state.new_session_key != None: #first time the session get saved
        st.session_state.session_index_tracker = st.session_state.new_session_key
        st.session_state.new_session_key = None
   
    #auto select the chat session to not a new session
    index = chat_sessions.index(st.session_state.session_index_tracker)   
    st.sidebar.selectbox("Select Chat Session", chat_sessions, key="session_key", index=index, on_change=track_index)
    st.sidebar.toggle("PDF Chat", key="pdf_chat", value=False)

    if st.session_state.session_key != "new_session":
        st.session_state.history = load_chat_history_json(config["chat_history_path"] + st.session_state.session_key)   
    else:
        st.session_state.history = []
    chat_history = StreamlitChatMessageHistory(key="history") 
    llm_chain = load_chain(chat_history)

    uploaded_pdf = st.sidebar.file_uploader("up;pad an pdf file", accept_multiple_files=True, key="pdf_upload", type=['pdf'], on_change=toggle_pdf_chat)
    
    if uploaded_pdf:
        with st.spinner("Processing pdf....."):
            add_documents_to_db(uploaded_pdf)

    user_input = st.text_input("Type your message here", key="user_input", on_change=set_send_input)
    send_button = st.button("Send   ", key="send_button")

    if send_button or st.session_state.send_input:
        if st.session_state.user_question != "":
            llm_chain = load_chain(chat_history)
            llm_response = llm_chain.run(st.session_state.user_question)
            st.session_state.user_question = ""

        st.session_state.send_input = False


    # Display chat history in UI
    if chat_history.messages != []:
            with chat_container:
                st.write("Chat History")
                for message in chat_history.messages:
                    st.chat_message(message.type).write(message.content)
    
    
    save_chat_history()

if __name__ == "__main__":
    main()