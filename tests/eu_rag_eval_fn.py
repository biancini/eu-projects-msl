""" Eval function for the RAG system in the EU project."""

from typing import Any, Optional

import os
import random

from openai import OpenAI
from pydantic import BaseModel, Field

from evals import Eval
from dotenv import load_dotenv

from euprojectsrag.rag_chain import RAGChain


class TestSampleExecution(BaseModel):
    """Class for Test Sample Execution."""

    score: int = Field(description="Score of the extraction, 1 to 10.")
    description: str = Field(description="Description of the extraction quality.")
    match: bool = Field(default=False, description="True if the actual answer is similar to the expected answer.")
    passed: Optional[bool] = Field(default=None, description="True if the test is passed.")
    refusal: Optional[bool] = Field(default=None, description="True if the model refused to answer the question.")


class LLMEval(Eval):
    """Eval class for the RAG"""

    def __init__(self, **kwargs):
        """
        Initializes the evaluation class.
        """
        super().__init__(**kwargs)
        self.name = "eu_rag_eval.v1"
        self.description = "Evaluate the quality of answers generated by the RAG system in the EU project."

        load_dotenv(override=True)

    def eval_sample(self, sample: Any, rng: random.Random):
        """
        Evaluates the quality of the actual answer against the expected answer using OpenAI's API.
        Returns a score from 1 to 10.
        """
        rag_chain = RAGChain()
        query = sample.get("input", "")
        project_name = rag_chain.project_name_extraction(query)[0][0]
        actual_answer = rag_chain.query_project(query, project_name)
        expected_answer = sample.get("ideal", "")
        pages = sample.get("pages", "")

        prompt = f"""
        You are a helpful assistant evaluating the quality of answers.
        
        Query: {query}
        
        Expected answer:
        {expected_answer}
        {pages}
        
        Actual answer: {actual_answer}

        On a scale from 1 to 10, how good is the actual answer in terms of relevance and completeness?
        Do a check also on the sources indicated, the one in the actual answer need to be included in the range of pages of the expected answer.
        Just respond with a json containing:
        - the score number as "score"
        - a "description" field with a text justifying the score
        - a "match" field that is true if the score is above 7.

        If you cannot asnwere the question, respond with a refusal message like "I cannot answer that.".

        Respond with a json object like this:
        {{
            "score": 8,
            "description": "The actual answer is relevant and complete, matching the expected answer.",
            "match": true,
        }}
        or this for an answer refusal:
        {{
            "score": 0,
            "description": "I cannot answer that.",
            "match": false,
        }}
        """

        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = client.beta.chat.completions.parse(
            model=os.getenv('MODEL'),
            messages=[{"role": "user", "content": prompt}],
            response_format=TestSampleExecution
        )

        message = response.choices[0].message.parsed

        message.refusal = message.description == "I cannot answer that."
        message.passed = not message.refusal and message.match

        return message.model_dump(mode='json')

    def run(self, recorder):
        samples = self.get_samples()
        scores = self.eval_all_samples(recorder, samples)

        execution = {}
        for cursample, cursscore in zip(samples, scores):
            execution['Test %d' % cursample['test_id']] = cursscore
        return execution
