from langgraph.constants import Send
from activevision_agent.states import (
    ActiveVisionOverallState,
    BatchSubgraphEntryState
)

def continue_to_subgraph(state: ActiveVisionOverallState) -> list[Send]:
    return_list = []
    for image in state.image:
        return_list.append(
            Send(
                    "batch_subgraph", 
                    BatchSubgraphEntryState(
                        image = image,
                        query = state.query
                    )
                )
        )
    return return_list