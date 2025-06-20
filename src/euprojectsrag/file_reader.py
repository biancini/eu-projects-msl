"""Module for processing and managing EU project documents and data."""

import logging
import os

from typing import List, Dict
from chromadb import PersistentClient

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

from langchain.schema import Document
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

        self.logger = logging.getLogger(__name__)

        self.croma_db_path = "./chroma_db"

        self.converter = DocumentConverter()
        self.chunker = HybridChunker(merge_peers=True)

        llm_provider = os.getenv('LLM_PROVIDER', 'google')

        embedding_model = os.getenv(f'{llm_provider.upper()}_EMBEDDING_MODEL')

        if llm_provider == 'openai':
            self.embedding = OpenAIEmbeddings(model=embedding_model)
        elif llm_provider == 'google':
            self.embedding = GoogleGenerativeAIEmbeddings(model=embedding_model)
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}. Supported providers are 'openai' and 'google'.")


    def get_documents_from_pdf(
            self,
            file_name: str,
            doc_type: str,
            project_name: str) -> Dict:
        """Get documents from a PDF file.

        Args:
            file_name: Path to the PDF file
            doc_type: Type of the document (call, proposal, ga)
            project_name: Name of the project

        Returns:
            List of documents
        """
        result = self.converter.convert(file_name)
        self.logger.info("Read document %s", file_name)
        documnent = {
            "document": result.document,
            "doc_type": doc_type,
            "project_name": project_name,
        }

        return documnent

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
        all_documents = [ call_docs, proposal_docs, ga_docs ]

        all_chunks = []
        for doc in all_documents:
            for chunk in self.chunker.chunk(doc["document"]):
                all_chunks.append({
                    "text": chunk.text,
                    "metadata": {
                        "source": chunk.meta.origin.filename,
                        "doc_type": doc['doc_type'],
                        "project_name": doc['project_name'],
                        "page_numbers": ', '.join([
                            str(page_no)
                            for page_no in sorted(
                                set(
                                    prov.page_no
                                    for item in chunk.meta.doc_items
                                    for prov in item.prov
                                )
                            )
                        ])
                        or None,
                        "title": chunk.meta.headings[0] if chunk.meta.headings else None,
                    },
                })

        documents: List[Document] = [
            Document(page_content=cur_chunk['text'], metadata=cur_chunk['metadata'])
            for cur_chunk in all_chunks
        ]
        return documents


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
