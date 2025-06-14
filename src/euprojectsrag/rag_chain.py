"""RAG (Retrieval-Augmented Generation) chain for EU project document analysis."""

import os
import logging

from typing import List, Dict

from openai import OpenAI
from dotenv import load_dotenv

from langchain.schema import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnableLambda
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from .data_models import ProjectExtraction, PROJECT_LIST
from .file_reader import FileReader
from .data_models import ProjectData
from .configurations import  get_project_conf


# pylint: disable=no-member

class RAGChain():
    """Class to handle the RAG chain for answering questions about EU projects."""

    def __init__(self):
        """Initialize the RAGChain."""
        self.logger = logging.getLogger(__name__)
        self.pdf_reader = FileReader()

        load_dotenv(override=True)
        self.memory = ConversationBufferMemory(return_messages=True)

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv('MODEL')
        self.logger.info("Using model: %s", self.model)


    def reciprocal_rank_fusion(self, results: List[List], max_docs: int = 400):
        """Reciprocal Rank Fusion that takes multiple lists of ranked documents and
        an optional parameter k used in the RRF formula.
        Uses document ID as key instead of serializing documents."""

        self.logger.info("Starting reciprocal rank fusion")
        fused_scores = {}
        doc_map = {}

        for docs in results:
            for rank, doc in enumerate(docs):
                doc_id = doc.id
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0
                    doc_map[doc_id] = doc
                fused_scores[doc_id] += 1. / (rank + 60)

        self.logger.info("Reciprocal rank fusion complete, processed %s documents",
                         len(fused_scores))

        fused_scores = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        if len(fused_scores) > max_docs:
            self.logger.info("Documents exceed max_docs (%s), truncating results", max_docs)
            fused_scores = fused_scores[:max_docs]

        reranked_results = [
            (doc_map[doc_id], score)
            for doc_id, score in fused_scores
        ]

        return reranked_results


    def run_rag(self, project_name: str, question: str, memory: ConversationBufferMemory) -> str:
        """Run RAG chain to answer questions about EU project documents.
        
        Args:
            project_name: Name of the project to query
            question: Question to ask about the project
            
        Returns:
            str: Answer to the question based on project documents
        """

        self.logger.info("Retrieve project configuration data")

        project_conf = get_project_conf(project_name)
        project_data = ProjectData(
            project_name=project_conf.project_name,
            start_date=project_conf.start_date,
        )

        self.logger.debug("Project configuration data: %s", project_conf)
        self.logger.debug("Project data: %s", project_data)

        self.logger.info("Retrieve project documents from PDF files")
        retriever = self.pdf_reader.read_pdf_files(project_conf)

        self.logger.info("Do RAG-fusion to improve document extraction")
        prompt_rag_fusion = ChatPromptTemplate.from_template("""
            You are a helpful assistant that generates multiple search queries based on a single input query. \n
            Generate multiple search queries related to: {question} \n
            Output (4 queries):
        """)

        self.logger.info("Generate queries")
        generate_queries = (
            prompt_rag_fusion
            | ChatOpenAI(model_name=os.getenv('MODEL'), temperature=0)
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )

        retrieval_chain_rag = generate_queries | retriever.map() | self.reciprocal_rank_fusion

        template = """You are a helpful assistant with access to the following context information:
            - Project Data: {context_project_data}
            - Project Documents: {context_project_docs}

            Based on the information above, please answer the following question as accurately and thoroughly as possible.
            If the information is not available in the context, say so explicitly but try to answer the question, if generic enough, based on the information available.

            Question: {question}

            Conversation history:
            {history}

            Please provide:
            A clear, plain-text answer (no markdown formatting).
            A list of all sources used, specifying the document name and page number(s) (e.g., "Proposal, p. 4"). If multiple documents are referenced, list them all.
            At the end of the response, include a "Sources" section. In this section, list only the unique page numbers referenced from the proposal.
            Sort them in ascending order and group consecutive or nearby pages (within 2-3 pages apart) into ranges.

            Format your response in JSON format and user Markdown formatting for the texts.
            Here an example of what you have to answer:
            {{
            'asnwer': 'Your detailed answer here',
            'sources': [
                {{'document_name': 'Call', 'page_numbers': '9, 10, 12-15'}},
                {{'document_name': 'Proposal', 'page_numbers': '10-14, 25, 32-34'}},
                {{'document_name': 'Grant Agreement', 'page_numbers': '1, 18-23'}}
            ]}}
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
                "context_project_docs": retrieval_chain_rag,
                "question": RunnablePassthrough(),
                "history": RunnableLambda(lambda _: memory.buffer)
            }
            | prompt
            | llm
            | JsonOutputParser()
        )

        self.logger.info("Invoke RAG chain")
        result = rag_chain.invoke(question)

        self.logger.info("Adding user question and AI response to memory")
        memory.chat_memory.add_user_message(question)
        memory.chat_memory.add_ai_message(result['answer'])

        return result


    def project_name_extraction(self, user_input: str) -> ProjectExtraction:
        """First LLM call to determine which project the user is asking about"""

        self.logger.info("Starting event extraction analysis")
        self.logger.debug("Input text: %s", user_input)

        human_messages = [
            msg.content
            for msg in self.memory.chat_memory.messages
            if isinstance(msg, HumanMessage)
        ]
        human_messages.append(user_input)

        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": f"""Analyze the query text history and answer if the \
                        query is related to a project.
                        Possible projects are these: {PROJECT_LIST}.
                        Return a list of the project names and for each project a confidence score between 0 and 1.
                        """,
                },
                {
                    "role": "user",
                    "content": '\n'.join(human_messages)
                },
            ],
            response_format=ProjectExtraction,
        )

        max_confidence = 0
        project_name = None
        result = completion.choices[0].message.parsed
        for (project, confidence) in zip(result.project_name, result.confidence_score):
            self.logger.info("Project found: %s with confidence %.2f", project, confidence)

            if (
                confidence >= 0.7
                and confidence > max_confidence
                ):
                max_confidence = confidence
                project_name = project

        if project_name is None:
            self.logger.info("No project found with sufficient confidence in the query")
            return "all"

        self.logger.info(
            "Extraction complete - Working on project name: %s, Confidence: %.2f",
            project_name,
            max_confidence
        )
        return project_name


    def get_working_project(self, user_input: str, project_name: str = "all") -> str:
        """Get the working project name based on user input.
        
        Args:
            user_input: The question or query from the user
            project_name: The name of the project to query
            
        Returns:
            str: The name of the project to work with"""

        self.logger.info("Extracting project name from user input")
        if project_name == "all":
            project_name = self.project_name_extraction(user_input)

        if project_name == "all":
            self.logger.info("No project found in the query, stopping processing")
            return None

        self.logger.info("Project name extraction complete, proceeding with event processing")
        return project_name


    def query_project(self, user_input: str, project_name: str) -> Dict[str, str]:
        """Query the project with the given question.
            Args:
            user_input: The question or query from the user
            project_name: The name of the project to query
            Returns:           
            str: The answer to the user's question based on the project documents"""

        self.logger.info("Processing user query")
        self.logger.debug("Raw input: %s", user_input)

        self.logger.info("Gate check passed, proceeding with event processing")
        query_result = self.run_rag(project_name, user_input, self.memory)
        query_result["project_name"] = project_name
        return query_result
