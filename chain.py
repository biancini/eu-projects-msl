"""RAG (Retrieval-Augmented Generation) chain for EU project document analysis."""

import os
import logging

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from datamodels import ProjectData
from confs import  get_project_conf
from readpdfs import read_pdf_files

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(results: list[list], k: int = 60):
    """Reciprocal Rank Fusion that takes multiple lists of ranked documents and
       an optional parameter k used in the RRF formula.
       Uses document ID as key instead of serializing documents."""

    logger.info("Starting reciprocal rank fusion")
    fused_scores = {}
    doc_map = {}
    max_docs = len(results[0])

    for docs in results:
        for rank, doc in enumerate(docs):
            doc_id = doc.id
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
                doc_map[doc_id] = doc
            fused_scores[doc_id] += 1. / (rank + k)

    logger.info("Reciprocal rank fusion complete, processed %s documents", len(fused_scores))

    fused_scores = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    if len(fused_scores) > max_docs:
        logger.info("More documents found than max_docs (%s), truncating results", max_docs)
        fused_scores = fused_scores[:max_docs]

    reranked_results = [
        (doc_map[doc_id], score)
        for doc_id, score in fused_scores
    ]

    return reranked_results


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
    retriever = read_pdf_files(project_conf)

    logger.info("Do RAG-fusion to improve document extraction")
    prompt_rag_fusion = ChatPromptTemplate.from_template("""
        You are a helpful assistant that generates multiple search queries based on a single input query. \n
        Generate multiple search queries related to: {question} \n
        Output (4 queries):
    """)

    logger.info("Generate queries")
    generate_queries = (
        prompt_rag_fusion
        | ChatOpenAI(model_name=os.getenv('MODEL'), temperature=0)
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )

    retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion

    template = """You are a helpful assistant with access to the following context information:
        - Project Data: {context_project_data}
        - Project Documents: {context_project_docs}

        Based on the information above, please answer the following question as accurately and thoroughly as possible.
        If the information is not available in the context, say so explicitly but try to answer the question, if generic enough, based on the information available.

        Question: {question}

        Please provide:
        A clear, plain-text answer (no markdown formatting).
        A list of all sources used, specifying the document name and page number(s) (e.g., "Proposal, p. 4"). If multiple documents are referenced, list them all.
        At the end of the response, include a "Sources" section. In this section, list only the unique page numbers referenced from the proposal.
        Sort them in ascending order and group consecutive or nearby pages (within 2-3 pages apart) into ranges.
        Use the format: Proposal, pp. 61, 68-70, 79.

        Format your response as follows and use Markdown formatting:
        ## Answer: 
        [Your detailed answer here]
        ## Sources:
        | Document Name   | Page Numbers     |
        | --------------- | ---------------- |
        | Call            | 9, 10, 12-15     |
        | Proposal        | 10-14, 25, 32-34 |
        | Grant Agreement | 1, 18-23         |
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model_name=os.getenv('MODEL'), temperature=0)

    context_project_data = {
        "project_name": project_data.project_name,
        "start_date": project_data.start_date,
    }

    rag_chain = (
        {
            "context_project_data": lambda _: context_project_data,
            "context_project_docs": retrieval_chain_rag_fusion,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info("Invoke RAG chain")
    result = rag_chain.invoke(question)
    return result
