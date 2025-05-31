"""Main module for EU project document analysis and query processing."""

import logging
import os

from termcolor import colored

from openai import OpenAI
from dotenv import load_dotenv

from datamodels import ProjectExtraction
from chain import run_rag

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = os.getenv('MODEL')

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def project_name_extraction(user_input: str) -> ProjectExtraction:
    """First LLM call to determine which project the user is asking about"""

    logger.info("Starting event extraction analysis")
    logger.debug("Input text: %s", user_input)

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": """Analyze the query text andasnwer if the query is related to a project.

                Possible proects are SPECTRO, EMAI4EU, RESCHIP4EU and ACHIEVE.
                """,
            },
            {"role": "user", "content": user_input},
        ],
        response_format=ProjectExtraction,
    )
    result = completion.choices[0].message.parsed
    logger.info(
        "Extraction complete - Project name: %s, Confidence: %.2f",
        result.project_name,
        result.confidence_score
    )
    return result


def query_project(user_input: str) -> str:
    """Query the project with the given question."""

    logger.info("Processing user query")
    logger.debug("Raw input: %s", user_input)

    initial_extraction = project_name_extraction(user_input)

    if (
        initial_extraction.project_name not in ["SPECTRO", "EMAI4EU", "RESCHIP4EU", "ACHIEVE"]
        or initial_extraction.confidence_score < 0.7
    ):
        logger.warning(
            "Gate check failed - project name: %s, confidence: %.2f",
            initial_extraction.project_name,
            initial_extraction.confidence_score
        )
        return None

    logger.info("Gate check passed, proceeding with event processing")

    query_result = run_rag(initial_extraction.project_name, user_input)
    return query_result


if __name__ == "__main__":
    question = input(colored("Enter the question you want to ask to the project:\n", "green"))
    asnwer = query_project(question)
    print(colored(asnwer, "green"))
