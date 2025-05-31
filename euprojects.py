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

logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CROMA_DB_PATH = "./chroma_db"


def read_pdf_pages(filename: str) -> list[str]:
    """Extract text from all pages of a PDF file.
    
    Args:
        filename: Path to the PDF file
        
    Returns:
        List of strings, where each string contains the text from one page
    """
    logger.info("Reading PDF pages for file %s", filename)
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

    logger.debug("Read %d pages", len(pages))
    return pages

def get_chroma_db(project_data: ProjectData, collection_name: str, pdf_filename: str) -> Chroma:
    """Get a Chroma object for a given collection name.
    
    Args:
        project_data: Project data object to store processed text
        collection_name: Name of the collection
        pdf_filename: Path to the PDF file
        
    Returns:
        Chroma object
    """

    logger.info("Getting Chroma DB client")
    croma_db_client = PersistentClient(path=CROMA_DB_PATH)
    collection_names = croma_db_client.list_collections()

    if collection_name in collection_names:
        logger.info("Collection %s already exists", collection_name)
        return Chroma(collection_name,
                      persist_directory="./chroma_db",
                      embedding_function=OpenAIEmbeddings(),
        )

    logger.info("Collection %s does not exist, creating it", collection_name)
    project_data.call_text = read_pdf_pages(pdf_filename)

    logger.info("Creating Chroma collection %s", collection_name)
    return Chroma.from_texts(collection_name=collection_name,
                                persist_directory="./chroma_db",
                                texts=project_data.call_text,
                                embedding=OpenAIEmbeddings(),
    )

def read_pdf_files(project_conf: ProjetFileData, project_data: ProjectData) -> Chroma:
    """Read and process PDF files for a project, including call, proposal,
    and grant agreement documents.
    
    Args:
        project_conf: Project configuration containing file paths
        project_data: Project data object to store processed text

    Returns:
        Chroma object
    """

    logger.info("Reading PDF files for project %s", project_conf.project_name)

    logger.info("Reading call file for project %s", project_conf.project_name)
    call_textdb = get_chroma_db(project_data,
                                project_conf.project_name + "_call",
                                project_conf.base_path + project_conf.call_file)

    logger.info("Reading proposal file for project %s", project_conf.project_name)
    proposal_textdb = get_chroma_db(project_data,
                                    project_conf.project_name + "_proposal",
                                    project_conf.base_path + project_conf.proposal_file)

    logger.info("Reading grant agreement file for project %s", project_conf.project_name)
    ga_textdb = get_chroma_db(project_data,
                              project_conf.project_name + "_ga",
                              project_conf.base_path + project_conf.ga_file)

    return call_textdb.as_retriever(), proposal_textdb.as_retriever(), ga_textdb.as_retriever()
