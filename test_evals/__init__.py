"""test_evals package"""

# Import necessary modules or classes here
from .eu_rag_completion_fn import EuRAGCompletionFn
from .eu_rag_eval_fn import LLMEval

__all__ = [
    "EuRAGCompletionFn",
    "LLMEval",
]

__version__ = "1.0.0"
__author__ = "Andrea Biancini <andrea.biancini@gmail.com>"
__license__ = "MIT"
__description__ = """
    Test evaluations for the eu-projects-msl package.
    This package includes functionalities for evaluating models in the context of European projects.
    """
