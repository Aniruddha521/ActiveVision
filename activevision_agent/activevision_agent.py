from langgraph.graph import StateGraph, START, END
from activevision_agent.states import (
    ActiveVisionEntryState,
    ActiveVisionOverallState,
    ActiveVisionOutputState
)
from activevision_agent.batch_subgraph import subgraph_builder
from activevision_agent.conditional_edges import human_interruption, continue_to_subgraph
from activevision_agent.nodes import (
    load_images,
    describe_query,
    display_output
)
activevision_builder = StateGraph(
    ActiveVisionOverallState,
    input = ActiveVisionEntryState,
    output = ActiveVisionOutputState
)

activevision_builder.add_node("load_images", load_images)
activevision_builder.add_node("describe_query", describe_query)
activevision_builder.add_node("batch_subgraph", subgraph_builder.compile())
activevision_builder.add_node("display_output", display_output)

activevision_builder.add_edge(START, "load_images")
activevision_builder.add_edge("load_images", "describe_query")
activevision_builder.add_conditional_edges(
    "describe_query", 
    continue_to_subgraph,
    ["batch_subgraph"]
    )
activevision_builder.add_edge("batch_subgraph", "display_output")
activevision_builder.add_conditional_edges(
    "display_output", 
    human_interruption,
    ["describe_query", END]
    )

activevision_graph = activevision_builder.compile()