from activevision_agent.states.image_state import ImageState

def image_reducer(left: ImageState |None, right: ImageState | None):
    if not right:
        return left
    return right