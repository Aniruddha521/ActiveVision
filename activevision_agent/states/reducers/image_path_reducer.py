def image_path_reducer(left: str | None, right: str | None):
    if not left:
        return right
    return left