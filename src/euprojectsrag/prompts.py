""" Module for generating prompts for the RAG system.
This module contains classes and methods to create prompts that can be used
in a Retrieval-Augmented Generation (RAG) system.
"""

from langchain.prompts import ChatPromptTemplate

class PromptsGenerator:
    """Class to generate prompts for the RAG system."""

    @staticmethod
    def get_ragfusion_prompt() -> ChatPromptTemplate:
        """ Generate a prompt for the RAG Fusion system.
        This prompt is used to instruct the model to provide relevant information
        based on the user's query.

        Returns:
            ChatPromptTemplate: A template for the RAG Fusion prompt.
        """

        return ChatPromptTemplate.from_template("""
            You are a helpful assistant that generates multiple search queries based on a single input query.
            Generate multiple search queries related to: {question}
            Output (4 queries):
        """)

    @staticmethod
    def get_projects_names_prompt() -> ChatPromptTemplate:
        """ Generate a prompt to retrieve project names.
        This prompt is used to instruct the model to provide a list of project names
        based on the user's query.

        Returns:
            ChatPromptTemplate: A template for the project names prompt.
        """

        return ChatPromptTemplate.from_template("""
            You are a smart assistant that helps identify which research projects are relevant to a given user query.

            Here is a list of possible project names: {PROJECT_LIST}

            Your task is to analyze the user query and return:
            - `project_names`: the list of project names from the list above that are clearly mentioned or strongly implied
            - `confidence_scores`: for each project listed, return a confidence score between 0.0 (not confident) and 10.0 (very confident)

            Only include projects if their presence is explicitly mentioned or reasonably inferable. 
            Be precise and avoid guessing.

            User query: {query}

            Output format:
            {format_instructions}
            """)

    @staticmethod
    def get_projects_details_prompt() -> ChatPromptTemplate:
        """ Generate a prompt to retrieve project details.
        This prompt is used to instruct the model to provide detailed information
        about the projects based on the user's query.

        Returns:
            ChatPromptTemplate: A template for the project details prompt.
        """

        return ChatPromptTemplate.from_template("""
            You are a helpful assistant with access to the following context information:
            - Project Data: {context_project_data}
            - Project Documents: {context_project_docs}

            Based on the information above, please answer the following question as accurately and thoroughly as possible.
            Answer by searching information in the Grant Agreement and if you can'f find the information there, search in the other documents.
            If the information is not available in the context, say so explicitly but try to answer the question, if generic enough, based on the information available.

            Question: {question}

            Conversation history:
            {history}

            Please provide:
            A clear answer in Markdown formatting.
            A list of all sources used, specifying the document name and page number(s) (e.g., "Proposal, p. 4, 5-10").
            List each document name only once, even if multiple pages are referenced and group them together.
            If multiple documents are referenced, list them all.
            At the end of the response, include a "Sources" section. In this section, list only the unique page numbers referenced from the proposal.
            Sort them in ascending order and group consecutive or nearby pages (within 2-3 pages apart) into ranges.

            Output format:
            {format_instructions}
        """)

    @staticmethod
    def get_projects_question_prompt_prompt() -> ChatPromptTemplate:
        """ Generate a prompt to summarize project information.
        This prompt is used to instruct the model to provide a summary of the project
        information based on the user's query.

        Returns:
            ChatPromptTemplate: A template for the project summary prompt.
        """

        return ChatPromptTemplate.from_template("""
            You have to answer this question from the user: {question}
            To answer it, you must retrieve information *exclusively* from the {project_name} project documents and data.

            Generate a query prompt that retrieves the specific information needed to answer the question,
            focused strictly on the {project_name} project.
            Do not include any references to other projects or comparative elements.

            The beginning of the prompt should be exactly:
            "You are a helpful assistant with access to the following context information:
            - Project Data: {{context_project_data}}
            - Project Documents: {{context_project_docs}}
            - Original Question: {{question}}"

            Only use the information provided in the first two variables.
            Ignore any other potential knowledge, including data from other projects.

            Please provide:
            The generated prompt query, and must be specific, concise, and limited to the {project_name} context.
            A list of all sources used, specifying the document name and page number(s) (e.g., "Proposal, p. 4, 7-10").
            If multiple documents are referenced, list them all. Group all pages from the same document together.
            At the end of the response, include a "Sources" section. In this section, list only the unique page numbers referenced from the proposal.
            Sort them in ascending order and group consecutive or nearby pages (within 2-3 pages apart) into ranges.

            Output format:
            {format_instructions}
        """)

    @staticmethod
    def get_multiple_projects_question_prompt() -> ChatPromptTemplate:
        """ Generate a prompt to summarize information from multiple projects.
        This prompt is used to instruct the model to provide a summary of the project
        information based on the user's query across multiple projects.

        Returns:
            ChatPromptTemplate: A template for the multi-project summary prompt.
        """

        return ChatPromptTemplate.from_template("""
            You are a helpful assistant that generates a query based on a user input.
            You will be asked to provide an answer comparing information from multiple projects.
                        
            Answer the following question based on the information provided from multiple projects:
            {question}

            You have to answer the question based on the information provided by the following projects:
            {projects}

            The information from each project is provided below:
            {projects_info}

            Provide a clear, concise answer in Markdown format that synthesizes the information from all projects.
            If there are conflicting pieces of information, explain the differences and provide a balanced view.

            Output format:
            {format_instructions}
        """)
