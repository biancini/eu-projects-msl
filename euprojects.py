"""Module for processing and managing EU project documents and data."""

import os
import pdfplumber
from openai import OpenAI

from langchain_chroma import Chroma

from langchain_openai import OpenAIEmbeddings


from models import ProjectData
from confs import ProjetFileData

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

    with pdfplumber.open(filename) as pdf:
        page_num = 1
        for page in pdf.pages:
            page_text = page.extract_text()
            pages.append(f"Page number {page_num}: {page_text}")
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

    if os.path.isdir(CROMA_DB_PATH):
        call_textdb = Chroma(collection_name=project_conf.project_name + "_call",
                              persist_directory="./chroma_db",
                              embedding_function=OpenAIEmbeddings(),
        )
        proposal_textdb = Chroma(collection_name=project_conf.project_name + "_proposal",
                                  persist_directory="./chroma_db",
                                  embedding_function=OpenAIEmbeddings(),
        )
        ga_textdb = Chroma(collection_name=project_conf.project_name + "_ga",
                            persist_directory="./chroma_db",
                            embedding_function=OpenAIEmbeddings(),
        )

        return call_textdb.as_retriever(), proposal_textdb.as_retriever(), ga_textdb.as_retriever()

    project_data.call_text=read_pdf_pages(project_conf.base_path + project_conf.call_file)
    project_data.proposal_text=read_pdf_pages(project_conf.base_path + project_conf.proposal_file)
    project_data.ga_text=read_pdf_pages(project_conf.base_path + project_conf.ga_file)

    call_textdb = Chroma.from_texts(collection_name=project_conf.project_name + "_call",
                                    persist_directory="./chroma_db",
                                    texts=project_data.call_text,
                                    embedding=OpenAIEmbeddings(),
    )

    proposal_textdb = Chroma.from_texts(collection_name=project_conf.project_name + "_proposal",
                                        persist_directory="./chroma_db",
                                        texts=project_data.proposal_text,
                                        embedding=OpenAIEmbeddings(),
    )

    ga_textdb = Chroma.from_texts(collection_name=project_conf.project_name + "_ga",
                                  persist_directory="./chroma_db",
                                  texts=project_data.ga_text,
                                  embedding=OpenAIEmbeddings(),
    )

    return call_textdb.as_retriever(), proposal_textdb.as_retriever(), ga_textdb.as_retriever()
