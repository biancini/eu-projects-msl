"""RAG (Retrieval-Augmented Generation) chain for EU project document analysis."""

import os
import logging

from typing import List, Dict, Tuple, Type
from dotenv import load_dotenv
from pydantic import BaseModel

from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from .data_models import PROJECT_LIST
from .file_reader import FileReader
from .prompts import PromptsGenerator
from .data_models import ProjectData, ProjectExtraction, LLMBasicAnswer, LLMAnswerWithSources
from .configurations import  get_project_conf

# pylint: disable=no-member

class RAGChain():
    """Class to handle the RAG chain for answering questions about EU projects."""

    def __init__(self):
        """Initialize the RAGChain."""
        self.logger = logging.getLogger(__name__)
        self.pdf_reader = FileReader()

        load_dotenv(override=True)

        llm_provider = os.getenv('LLM_PROVIDER', 'google')

        self.model = os.getenv(f'{llm_provider.upper()}_MODEL')

        if llm_provider == 'openai':
            self.llm = ChatOpenAI(model_name=self.model, temperature=0)
        elif llm_provider == 'google':
            self.llm = ChatGoogleGenerativeAI(model=self.model, temperature=0.0,)
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}. Supported providers are 'openai' and 'google'.")
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


    def call_llm(
            self,
            rag_params: Dict,
            prompt: ChatPromptTemplate,
            response_type: Type[BaseModel] = LLMBasicAnswer) -> BaseModel:
        """Call the LLM with the given parameters and prompt.
        
        Args:
            rag_params: Parameters for the RAG chain
            prompt: Prompt template to be used in the LLM
            question: Question to ask the LLM
            response_type: Type of the response to be returned, if None, returns a string

        Returns:
            BaseModel: The response from the LLM, formatted as specified by response_type
        """

        parser = PydanticOutputParser(pydantic_object=response_type)
        prompt = prompt.partial(format_instructions=parser.get_format_instructions())

        self.logger.info("Send call to LLM")
        rag_chain = rag_params | prompt | self.llm | parser
        result = rag_chain.invoke("")

        return result


    def run_rag(
            self,
            project_name: str,
            prompt: str,
            question: str,
            memory: List) -> str:
        """Run RAG chain to answer questions about EU project documents.
        
        Args:
            project_name: Name of the project to query
            prompt_template: Template for the prompt to be used in the LLM
            question: Question to ask about the project
            memory: Whether to use memory for the conversation history
            
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
        prompt_rag_fusion = PromptsGenerator.get_ragfusion_prompt()

        self.logger.info("Creating RAG chain with prompt and LLM")
        retrieval_chain_rag = (
            prompt_rag_fusion
            | self.llm
            | StrOutputParser()
            | (lambda x: x.split("\n"))
            | retriever.map()
            | self.reciprocal_rank_fusion
        )

        context_project_data = {
            "project_name": project_data.project_name,
            "start_date": project_data.start_date,
        }

        rag_params = {
            "context_project_data": lambda _: context_project_data,
            "context_project_docs": retrieval_chain_rag,
            "question": lambda _: question
        }
        if memory and len(memory) > 0:
            rag_params["history"] = RunnableLambda(lambda _: [msg["content"] for msg in memory if msg["role"] == "user"])

        self.logger.info("Invoke RAG chain")
        result = self.call_llm(
            rag_params,
            prompt,
            response_type=LLMAnswerWithSources,
        )

        return result


    def project_name_extraction(self, messages: List) -> List[Tuple[str, float]]:
        """LLM call to determine which project the user is asking about.
        Args:
            user_input: The question or query from the user
        Returns:
            str: The list of project names extracted from the user input
        """

        user_input = messages[-1]["content"]

        self.logger.info("Starting event extraction analysis")
        self.logger.debug("Input text: %s", user_input)

        human_messages = [
            msg["content"]
            for msg in messages
            if msg["role"] == "user"
        ]

        prompt = PromptsGenerator.get_projects_names_prompt()

        result = self.call_llm(
            {
                "PROJECT_LIST": lambda _: ", ".join(PROJECT_LIST),
                "query": lambda _: '\n'.join(human_messages)
            },
            prompt,
            response_type=ProjectExtraction,
        )

        projects = [
            (project, confidence) for project, confidence
            in zip(result.project_names, result.confidence_scores)
            if confidence >= 0.5
        ]

        projects = sorted(projects, key=lambda x: x[1], reverse=True)
        if len(projects) == 0:
            self.logger.info("No project found with sufficient confidence in the query")
            return [("all", 1.0)]

        self.logger.info("Extracted projecs: %s", projects)
        return projects


    def get_working_project(self, user_input: str, project_name: str = "all") -> str:
        """Get the working project name based on user input.
        
        Args:
            user_input: The question or query from the user
            project_name: The name of the project to query
            
        Returns:
            str: The name of the project to work with"""

        self.logger.info("Extracting project name from user input")
        if project_name == "all":
            project_name = self.project_name_extraction(user_input)[0][0]

        if project_name == "all":
            self.logger.info("No project found in the query, stopping processing")
            return None

        self.logger.info("Project name extraction complete, proceeding with event processing")
        return project_name


    def query_project(self, messages: List, project_name: str) -> Dict[str, str]:
        """Query the project with the given question.

        Args:
            user_input: The question or query from the user
            project_name: The name of the project to query
        
        Returns:           
            str: The answer to the user's question based on the project documents"""
        
        user_input = messages[-1]["content"]

        self.logger.info("Processing user query")
        self.logger.debug("Raw input: %s", user_input)

        prompt = PromptsGenerator.get_projects_details_prompt()

        self.logger.info("Gate check passed, proceeding with event processing")
        query_result = self.run_rag(project_name, prompt, user_input, messages)
        query_result.sources = [
            {
                'document_name': project_name + " " + source['document_name'],
                'page_numbers': source['page_numbers']
            } for source in query_result.sources
        ]

        return query_result


    def generate_query_per_project(self, user_input: str, project_name: str) -> str:
        """Generate a list of project names based on the user input.
        
        Args:
            user_input: The question or query from the user
            project_name: The name of the project to query
            
        Returns:
            str: The query to be used to retrieve information from the project documents
        """

        self.logger.info("Extracting answer about project %s", project_name)
        prompt = PromptsGenerator.get_projects_question_prompt_prompt()

        result = self.call_llm(
            {
                "project_name": lambda _: project_name,
                "question": lambda _: user_input,
            },
            prompt,
            response_type=LLMAnswerWithSources,
        )
        return result


    def query_projects(self, messages: List, project_names: List[str]) -> str:
        """Query the project with the given question.
        
        Args:
            user_input: The question or query from the user
            project_names: The names of the projects to query
        
        Returns:           
            str: The answer to the user's question based on the project documents"""

        user_input = messages[-1]["content"]

        self.logger.info("Processing user query")
        self.logger.debug("Raw input: %s", user_input)

        query_results = {}
        sources = []

        for project_name in [p[0] for p in project_names]:
            self.logger.info("Processing project: %s", project_name)

            generated_query = self.generate_query_per_project(user_input, project_name)
            for source in generated_query.sources:
                sources.append({
                    'document_name': project_name + " " + source['document_name'],
                    'page_numbers': source['page_numbers']
                })

            prompt_template = ChatPromptTemplate.from_template(generated_query.answer + """

                Output format:
                {format_instructions}""")

            result = self.run_rag(project_name, prompt_template, user_input)
            query_results[project_name] = result.answer

        self.logger.info("Obtained query results for all projects")

        prompt = PromptsGenerator.get_multiple_projects_question_prompt()

        result = self.call_llm(
            {
                "projects": lambda _: ', '.join(query_results.keys()),
                "projects_info": lambda _: ', '.join([str(result) for result in query_results.items()]),
                "question": lambda _: user_input
            },
            prompt,
            response_type=LLMBasicAnswer,
        )

        return LLMAnswerWithSources(answer=result.answer, sources=sources)
