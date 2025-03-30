from activevision_agent.states import ActiveVisionOverallState
from langgraph.graph import END

def human_interruption(state: ActiveVisionOverallState):
    if state.human_feedback == "":
        return END
    return "display_output"
