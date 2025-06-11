"""Generates a streamlit web interface to chat with the bot, with logs shown on the right."""

import logging

import urllib.parse

import streamlit as st

from euprojectsrag.rag_chain import RAGChain
from euprojectsrag.file_reader import FileReader
from euprojectsrag.configurations import ProjetFileData
from euprojectsrag.configurations import get_project_conf

st.set_page_config(
    page_title="RAG Chat System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "current_collection" not in st.session_state:
    st.session_state.current_collection = "default"

def main():
    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    st.title("🤖 EU Projects Answering Bot")
    st.markdown("Chat with your documents using AI-powered retrieval and generation")

    rag_chain = RAGChain()
    reader = FileReader()

    with st.sidebar:
        st.header("📚 Document Management")

        # Collection selection
        collections = reader.get_collection_names()
        coll_names = ["default"]
        coll_names += collections.keys()

        selected_collection = st.selectbox(
            "Select Project",
            coll_names,
            index=coll_names.index(st.session_state.current_collection) if st.session_state.current_collection in coll_names else 0
        )
        if selected_collection != st.session_state.current_collection:
            st.session_state.current_collection = selected_collection

        st.divider()
        
        # New collection
        with st.expander("Create New Project", expanded=False):
            new_collection = st.text_input("Project name")
            start_date = st.date_input("Start date")

            with st.container():
                call_file = st.file_uploader("Call PDF document path", type="pdf", accept_multiple_files=False)
                proposal_file = st.file_uploader("Proposal PDF document path", type="pdf", accept_multiple_files=False)
                ga_file = st.file_uploader("Grant Agreement PDF document path", type="pdf", accept_multiple_files=False)


            if st.button("Create Project") and new_collection and start_date and call_file and proposal_file and ga_file:
                project_conf = ProjetFileData(
                    project_name=new_collection,
                    start_date=start_date.format("%Y-%m-%d"),
                    base_path="",
                    call_file=call_file.path,
                    proposal_file=proposal_file.path,
                    ga_file=ga_file.path,
                )
                
                reader.read_project_files(project_conf)

        st.divider()
        
        # Collection info
        if st.session_state.current_collection:
            doc_count = collections[st.session_state.current_collection] if st.session_state.current_collection in collections else sum(collections.values())
            st.info(f"Collection: {st.session_state.current_collection} | Documents: {doc_count}")
        
        # Clear conversation
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.session_state.conversation_memory.clear()
            st.rerun()

    chat_container = st.container()
    with chat_container:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask your question..."):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                project_name, response = rag_chain.query_project(prompt, selected_collection)
                project_conf = get_project_conf(project_name)
                st.markdown(response['answer'])

                with st.expander("📄 Sources"):
                    sources_list = "<table><tr><td><b>Document Name</b></td><td><b>Page Numbers</b></td></tr>"

                    for document in response['sources']:
                        if 'Call' == document['document_name']:
                            url = urllib.parse.quote(project_conf.base_path + project_conf.call_file)
                        elif 'Proposal' == document['document_name']:
                            url = urllib.parse.quote(project_conf.base_path + project_conf.proposal_file)
                        elif 'Grant Agreement' == document['document_name']:
                            url = urllib.parse.quote(project_conf.base_path + project_conf.ga_file)
                        else:
                            url = None

                        if url is None:
                            sources_list += "<tr><td>" + document['document_name'] + "</td><td> " + \
                                document['page_numbers'] + "</td></td>"
                        else:
                            sources_list += f"<tr><td><a href=\"file://{url}\" target=\"_blank\">" + \
                                document['document_name'] + "</td><td>" + \
                                document['page_numbers'] + " </td></tr>"

                    sources_list += "</table>"
                    st.markdown(sources_list, unsafe_allow_html=True)

                st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
