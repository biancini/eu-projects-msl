"""Module for processing and managing EU project documents and data."""

import os
import logging
import pdfplumber
from openai import OpenAI

from chromadb import PersistentClient

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from models import ProjectData
from confs import ProjetFileData

logging.getLogger("pdfminer").setLevel(logging.ERROR)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CROMA_DB_PATH = "./chroma_db"


def read_pdf_pages(filename: str) -> list[str]:
    """Extract text from all pages of a PDF file.
    
    Args:
        filename: Path to the PDF file
        
    Returns:
        List of strings, where each string contains the text from one page
    """
    pages = []

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300,
        chunk_overlap=50)

    with pdfplumber.open(filename) as pdf:
        page_num = 1
        for page in pdf.pages:
            page_text = page.extract_text()
            splits = text_splitter.split_text(page_text)
            for split in splits:
                pages.append(f"Page number {page_num}: {split}")
            page_num += 1

    return pages

def read_pdf_files(project_conf: ProjetFileData, project_data: ProjectData) -> Chroma:
    """Read and process PDF files for a project, including call, proposal,
    and grant agreement documents.
    
    Args:
        project_conf: Project configuration containing file paths
        project_data: Project data object to store processed text

    Returns:
        Chroma object
    """

    croma_db_client = PersistentClient(path=CROMA_DB_PATH)
    collection_names = [col.name for col in croma_db_client.list_collections()]

    if project_conf.project_name + "_call" in collection_names:
        call_textdb = Chroma(collection_name=project_conf.project_name + "_call",
                             persist_directory="./chroma_db",
                             embedding_function=OpenAIEmbeddings(),
        )
    else:
        project_data.call_text=read_pdf_pages(project_conf.base_path + project_conf.call_file)

        call_textdb = Chroma.from_texts(collection_name=project_conf.project_name + "_call",
                                        persist_directory="./chroma_db",
                                        texts=project_data.call_text,
                                        embedding=OpenAIEmbeddings(),
        )

    if project_conf.project_name + "_proposal" in collection_names:
        proposal_textdb = Chroma(collection_name=project_conf.project_name + "_proposal",
                                 persist_directory="./chroma_db",
                                 embedding_function=OpenAIEmbeddings(),
        )
    else:
        project_data.proposal_text=read_pdf_pages(project_conf.base_path + project_conf.proposal_file)

        proposal_textdb = Chroma.from_texts(collection_name=project_conf.project_name + "_proposal",
                                            persist_directory="./chroma_db",
                                            texts=project_data.proposal_text,
                                            embedding=OpenAIEmbeddings(),
        )

    if project_conf.project_name + "_ga" in collection_names:
        ga_textdb = Chroma(collection_name=project_conf.project_name + "_ga",
                           persist_directory="./chroma_db",
                           embedding_function=OpenAIEmbeddings(),
        )
    else:
        project_data.ga_text=read_pdf_pages(project_conf.base_path + project_conf.ga_file)

        ga_textdb = Chroma.from_texts(collection_name=project_conf.project_name + "_ga",
                                      persist_directory="./chroma_db",
                                      texts=project_data.ga_text,
                                      embedding=OpenAIEmbeddings(),
        )

    return call_textdb.as_retriever(), proposal_textdb.as_retriever(), ga_textdb.as_retriever()
