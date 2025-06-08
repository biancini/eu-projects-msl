"""Module for processing and managing EU project documents and data."""

import os
import logging

import fitz
from typing import List

from openai import OpenAI
from chromadb import PersistentClient

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from euprojectsrag.confs import ProjetFileData

logging.getLogger("pdfminer").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CROMA_DB_PATH = "./chroma_db"



def extract_text_by_page(pdf_path):
    """Extract text from each page of a PDF file.
        Args:
        pdf_path: Path to the PDF file
    Returns:
        List of strings, where each string contains the text from one page"""

    logger.info("Extracting text from PDF file %s", pdf_path)

    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc]

    logger.debug("Extracted %d pages from PDF", len(pages))
    return pages

def read_pdf_pages(filename: str) -> List[str]:
    """Extract text from all pages of a PDF file.
    
    Args:
        filename: Path to the PDF file
        
    Returns:
        List of strings, where each string contains the text from one page
    """
    logger.info("Reading PDF pages for file %s", filename)

    doc = fitz.open(filename)
    pages = [page.get_text() for page in doc]

    logger.debug("Read %d pages", len(pages))
    return pages


def get_documents_from_pdf(file_name: str, doc_type: str, project_name: str) -> List[Document]:
    """Get documents from a PDF file.

    Args:
        file_name: Path to the PDF file
        doc_type: Type of the document (call, proposal, ga)
        project_name: Name of the project

    Returns:
        List of documents
    """
    page_texts = read_pdf_pages(file_name)

    documents = [
        Document(
            id=hash(page_texts[i]),
            page_content=page_texts[i],
            metadata={
                "doc_type": doc_type,
                "page_number": i + 1,
                "project_name": project_name,
            }
        )
        for i in range(len(page_texts))
    ]

    logger.info("Extracted %d documents from %s", len(documents), file_name)
    return documents

def get_chroma_db(project_conf: ProjetFileData) -> Chroma:
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
    collections = croma_db_client.list_collections()

    col_name = project_conf.project_name

    if col_name in collections:
        logger.info("Collection %s already exists", col_name)
        return Chroma(
            collection_name=col_name,
            persist_directory="./chroma_db",
            embedding_function=OpenAIEmbeddings(),
        )

    logger.info("Collection %s does not exist, creating it", col_name)

    logger.info("Reading call file for project %s", project_conf.project_name)
    call_docs = get_documents_from_pdf(
        project_conf.base_path + project_conf.call_file,
        "Call",
        project_conf.project_name
    )

    logger.info("Reading proposal file for project %s", project_conf.project_name)
    proposal_docs = get_documents_from_pdf(
        project_conf.base_path + project_conf.proposal_file,
        "Proposal",
        project_conf.project_name
    )

    logger.info("Reading GA file for project %s", project_conf.project_name)
    ga_docs = get_documents_from_pdf(
        project_conf.base_path + project_conf.ga_file,
        "Grant Agreement",
        project_conf.project_name
    )

    logger.info("Splitting documents")
    documents = call_docs + proposal_docs + ga_docs
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(documents)

    logger.info("Creating Chroma collection %s", col_name)
    return Chroma.from_documents(collection_name=col_name,
                                 persist_directory="./chroma_db",
                                 documents=split_docs,
                                 embedding=OpenAIEmbeddings(),
    )


def read_pdf_files(project_conf: ProjetFileData, returned_size : int = 100) -> Chroma:
    """Read and process PDF files for a project, including call, proposal,
    and grant agreement documents.
    
    Args:
        project_conf: Project configuration containing file paths
        returned_size: Number of documents to return from the retriever

    Returns:
        Chroma object
    """

    logger.info("Reading PDF files for project %s", project_conf.project_name)

    textdb = get_chroma_db(project_conf)
    retriever = textdb.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": returned_size,
            "fetch_k": int(returned_size*1.2),
            "lambda_mult": 0.8,
        }
    )

    logger.info("Retriever created with k=%s", returned_size)
    return retriever
