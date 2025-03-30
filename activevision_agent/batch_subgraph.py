from langgraph.graph import StateGraph, START, END
from activevision_agent.states import (
    BatchSubgraphEntryState,
    BatchSubgraphOverallState,
    BatchSubgraphOutputState
)
from activevision_agent.conditional_edges import handle_review
from activevision_agent.nodes import generate_output, review_output

subgraph_builder = StateGraph(
    BatchSubgraphOverallState,

    input = BatchSubgraphEntryState,
    output = BatchSubgraphOutputState
    )


subgraph_builder.add_node("generate_batch_response",generate_output)

subgraph_builder.add_node("review_batch_response",review_output)


subgraph_builder.add_edge(START, "generate_batch_response")

subgraph_builder.add_edge("generate_batch_response", "review_batch_response")
subgraph_builder.add_conditional_edges(
    "review_batch_response",
    handle_review,
    ["generate_batch_response", END]
)