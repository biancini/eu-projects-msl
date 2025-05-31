"""RAG (Retrieval-Augmented Generation) chain for EU project document analysis."""

import os
import logging

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from models import ProjectData
from confs import  get_project_conf
from euprojects import read_pdf_files

logger = logging.getLogger(__name__)

def run_rag(project_name: str, question: str) -> str:
    """Run RAG chain to answer questions about EU project documents.
    
    Args:
        project_name: Name of the project to query
        question: Question to ask about the project
        
    Returns:
        str: Answer to the question based on project documents
    """

    logger.info("Retrieve project configuration data")

    project_conf = get_project_conf(project_name)
    project_data = ProjectData(
        project_name=project_conf.project_name,
        start_date=project_conf.start_date,
    )

    logger.debug("Project configuration data: %s", project_conf)
    logger.debug("Project data: %s", project_data)

    logger.info("Retrieve project documents from PDF files")
    vectorstore_call, vectorstore_proposal, vectorstore_ga = read_pdf_files(project_conf, project_data)

    logger.info("Build RAG chain")
    template = """You are a helpful assistant with access to the following context information:
    - Project Data: {context_project_data}
    - Call Text: {context_call}
    - Project Proposal: {context_proposal}
    - Grant Agreement: {context_ga}

    Based on the information above, please answer the following question as accurately and thoroughly as possible.
    If the information is not available in the context, say so explicitly.

    Question: {question}

    Please provide:
    A clear, plain-text answer (no markdown formatting).
    A list of all sources used, specifying the document name and page number(s) (e.g., "Proposal, p. 4"). If multiple documents are referenced, list them all.

    Format your response as follows:
    Answer:
    [Your detailed answer here]
    Sources:
    - Call Text, p. X
    - Proposal, p. Y-Z
    - Grant Agreement, p. N

    If the answer is not found in the documents, respond:
    Answer: The information requested is not available in the provided context.
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model_name=os.getenv('MODEL'), temperature=0)

    logger.info("Invoke RAG chain")

    context_project_data = {
        "project_name": project_data.project_name,
        "start_date": project_data.start_date,
    }

    rag_chain = (
        {
            "context_call": vectorstore_call,
            "context_proposal": vectorstore_proposal,
            "context_ga": vectorstore_ga,
            "context_project_data": lambda _: context_project_data,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    result = rag_chain.invoke(question)
    return result
