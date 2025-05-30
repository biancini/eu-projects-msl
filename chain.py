"""RAG (Retrieval-Augmented Generation) chain for EU project document analysis."""

import os

from dotenv import load_dotenv
from langchain import hub
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from models import ProjectData
from confs import  get_project_conf
from euprojects import read_pdf_files

load_dotenv()

QUESTION = input("Enter the question you want to ask to the project:\n")
#QUESTION = "List all the project's KPIs."

PROJECT_NAME = "SPECTRO"
project_conf = get_project_conf(PROJECT_NAME)
project_data = ProjectData(
    project_name=project_conf.project_name,
    start_date=project_conf.start_date,
)

vectorstore_call, vectorstore_proposal, vectorstore_ga = read_pdf_files(project_conf, project_data)

docs_call = vectorstore_call.invoke(QUESTION)
docs_proposal = vectorstore_proposal.invoke(QUESTION)
docs_ga = vectorstore_ga.invoke(QUESTION)

TEMPLATE = """You are a helpful assistant with access to the following context information:
– Call Text: {context_call}
– Project Proposal: {context_proposal}
– Grant Agreement: {context_ga}
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
– Call Text, p. X
– Proposal, p. Y-Z
– Grant Agreement, p. N
If the answer is not found in the documents, respond:
Answer: The information requested is not available in the provided context.
"""

prompt = ChatPromptTemplate.from_template(TEMPLATE)
llm = ChatOpenAI(model_name=os.getenv('MODEL'), temperature=0)
chain = prompt | llm
chain.invoke({
    "context_call": docs_call,
    "context_proposal": docs_proposal,
    "context_ga": docs_ga,
    "question": QUESTION
})

prompt_hub_rag = hub.pull("rlm/rag-prompt")

rag_chain = (
    {
        "context_call": vectorstore_call,
        "context_proposal": vectorstore_proposal,
        "context_ga": vectorstore_ga,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

result = rag_chain.invoke(QUESTION)
print(result)
