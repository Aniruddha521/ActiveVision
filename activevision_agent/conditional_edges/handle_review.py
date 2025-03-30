from activevision_agent.states import BatchSubgraphOverallState
from langgraph.graph import END

def handle_review(state: BatchSubgraphOverallState):

    if state.max_review<=0 or state.proofreader_approval:
        return END
    return "generate_batch_response"