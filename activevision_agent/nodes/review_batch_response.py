from activevision_agent.states import BatchSubgraphOverallState
import dspy
from dotenv import load_dotenv
from activevision_agent.dspy_signatures import BatchOutputReviewerSignature

review_batch_email = dspy.ChainOfThought(BatchOutputReviewerSignature)

def review_output(state: BatchSubgraphOverallState) -> BatchSubgraphOverallState:
    pass