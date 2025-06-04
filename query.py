"""Main module for EU project document analysis and query processing."""

import logging
import os

from termcolor import colored

from langchain.schema import HumanMessage

from openai import OpenAI
from dotenv import load_dotenv
from datamodels import ProjectExtraction

from chain import run_rag, create_memory


load_dotenv(override=True)
memory = create_memory()

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = os.getenv('MODEL')
logger.info("Using model: %s", model)


def project_name_extraction(user_input: str) -> ProjectExtraction:
    """First LLM call to determine which project the user is asking about"""

    logger.info("Starting event extraction analysis")
    logger.debug("Input text: %s", user_input)

    human_messages = [msg.content for msg in memory.chat_memory.messages if isinstance(msg, HumanMessage)]
    human_messages.append(user_input)

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": """Analyze the query text history and answer if the query is related to a project.
                Possible projects are SPECTRO, EMAI4EU, RESCHIP4EU and ACHIEVE.
                Return a list of the project names and for each project a confidence score between 0 and 1.
                """,
            },
            {
                "role": "user",
                "content": '\n'.join([message for message in human_messages])
            },
        ],
        response_format=ProjectExtraction,
    )

    max_confidence = 0
    project_name = None
    result = completion.choices[0].message.parsed
    for (project, confidence) in zip(result.project_name, result.confidence_score):
        logger.info("Project found: %s with confidence %.2f", project, confidence)

        if (
            confidence >= 0.7
            and confidence > max_confidence
            ):
            max_confidence = confidence
            project_name = project

    if project_name is None:
        logger.info("No project found with sufficient confidence in the query")
    else:
        logger.info(
            "Extraction complete - Working on project name: %s, Confidence: %.2f",
            project_name,
            max_confidence
        )
    return project_name


def query_project(user_input: str) -> str:
    """Query the project with the given question."""

    logger.info("Processing user query")
    logger.debug("Raw input: %s", user_input)

    project_name = project_name_extraction(user_input)

    if project_name is None:
        logger.info("No project found in the query, stopping processing",)
        return None

    logger.info("Gate check passed, proceeding with event processing")
    query_result = run_rag(project_name, user_input, memory)
    return query_result


if __name__ == "__main__":
    question = input(colored("Enter the question you want to ask to the project:\n", "green"))
    asnwer = query_project(question)
    print(colored(asnwer, "green"))
