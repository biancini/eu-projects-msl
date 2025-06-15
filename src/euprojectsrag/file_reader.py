"""Module for processing and managing EU project documents and data."""

import logging
import os

from typing import List, Dict
import fitz
from chromadb import PersistentClient

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .configurations import ProjetFileData

class FileReader():
    """Class to handle reading and processing PDF files for EU projects.
    This class provides methods to extract text from PDF files, split the text into
    manageable chunks, and store the processed data in a Chroma database for further
    retrieval and analysis."""

    def __init__(self):
        logging.getLogger("pdfminer").setLevel(logging.ERROR)
        self.logger = logging.getLogger(__name__)

        self.croma_db_path = "./chroma_db"

        llm_provider = os.getenv('LLM_PROVIDER', 'google')

        embedding_model = os.getenv(f'{llm_provider.upper()}_EMBEDDING_MODEL')

        if llm_provider == 'openai':
            self.embedding = OpenAIEmbeddings(model=embedding_model)
        elif llm_provider == 'google':
            self.embedding = GoogleGenerativeAIEmbeddings(model=embedding_model)
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}. Supported providers are 'openai' and 'google'.")        

    def extract_text_by_page(self, pdf_path):
        """Extract text from each page of a PDF file.
            Args:
            fi
            pdf_path: Path to the PDF file
        Returns:
            List of strings, where each string contains the text from one page"""

        self.logger.info("Extracting text from PDF file %s", pdf_path)

        doc = fitz.open(pdf_path)
        pages = [page.get_text() for page in doc]

        self.logger.debug("Extracted %d pages from PDF", len(pages))
        return pages

    def read_pdf_pages(self, filename: str) -> List[str]:
        """Extract text from all pages of a PDF file.
        
        Args:
            filename: Path to the PDF file
            
        Returns:
            List of strings, where each string contains the text from one page
        """
        self.logger.info("Reading PDF pages for file %s", filename)

        doc = fitz.open(filename)
        pages = [page.get_text() for page in doc]

        self.logger.debug("Read %d pages", len(pages))
        return pages


    def get_documents_from_pdf(
            self,
            file_name: str,
            doc_type: str,
            project_name: str) -> List[Document]:
        """Get documents from a PDF file.

        Args:
            file_name: Path to the PDF file
            doc_type: Type of the document (call, proposal, ga)
            project_name: Name of the project

        Returns:
            List of documents
        """
        page_texts = self.read_pdf_pages(file_name)

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

        self.logger.info("Extracted %d documents from %s", len(documents), file_name)
        return documents

    def get_collection_names(self) -> Dict[str, int]:
        """Get the names of all collections in the Chroma database.
        
        Returns:
            List of collection names
        """
        croma_db_client = PersistentClient(path=self.croma_db_path)
        coll_names = croma_db_client.list_collections()

        collections = {}
        for coll in coll_names:
            doc_count = croma_db_client.get_collection(coll).count()
            collections[coll] = doc_count

        return collections


    def read_project_files(self, project_conf: ProjetFileData) -> List[Document]:
        """Read and process project files including call, proposal, and GA documents.
        Args:
            project_conf: Project configuration containing file paths
        Returns:
            List of processed documents
        """
        self.logger.info("Reading call file for project %s", project_conf.project_name)
        call_docs = self.get_documents_from_pdf(
            project_conf.base_path + project_conf.call_file,
            "Call",
            project_conf.project_name
        )

        self.logger.info("Reading proposal file for project %s", project_conf.project_name)
        proposal_docs = self.get_documents_from_pdf(
            project_conf.base_path + project_conf.proposal_file,
            "Proposal",
            project_conf.project_name
        )

        self.logger.info("Reading GA file for project %s", project_conf.project_name)
        ga_docs = self.get_documents_from_pdf(
            project_conf.base_path + project_conf.ga_file,
            "Grant Agreement",
            project_conf.project_name
        )

        self.logger.info("Splitting documents")
        documents = call_docs + proposal_docs + ga_docs
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        return splitter.split_documents(documents)


    def get_chroma_db(self, project_conf: ProjetFileData) -> Chroma:
        """Get a Chroma object for a given collection name.
        
        Args:
            project_data: Project data object to store processed text
            collection_name: Name of the collection
            pdf_filename: Path to the PDF file
            
        Returns:
            Chroma object
        """

        self.logger.info("Getting Chroma DB client")
        croma_db_client = PersistentClient(path=self.croma_db_path)
        collections = croma_db_client.list_collections()

        col_name = project_conf.project_name

        if col_name in collections:
            self.logger.info("Collection %s already exists", col_name)
            return Chroma(
                collection_name=col_name,
                persist_directory="./chroma_db",
                embedding_function=self.embedding,
            )

        self.logger.info("Collection %s does not exist, creating it", col_name)
        split_docs = self.read_project_files(project_conf)

        self.logger.info("Creating Chroma collection %s", col_name)
        return Chroma.from_documents(collection_name=col_name,
                                    persist_directory="./chroma_db",
                                    documents=split_docs,
                                    embedding=self.embedding,
        )


    def read_pdf_files(self, project_conf: ProjetFileData, returned_size : int = 500) -> Chroma:
        """Read and process PDF files for a project, including call, proposal,
        and grant agreement documents.
        
        Args:
            project_conf: Project configuration containing file paths
            returned_size: Number of documents to return from the retriever

        Returns:
            Chroma object
        """

        self.logger.info("Reading PDF files for project %s", project_conf.project_name)

        textdb = self.get_chroma_db(project_conf)
        retriever = textdb.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": returned_size,
                "fetch_k": int(returned_size*1.2),
                "lambda_mult": 0.8,
            }
        )

        self.logger.info("Retriever created with k=%s", returned_size)
        return retriever
