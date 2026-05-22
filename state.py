from typing import TypedDict, List, Optional

class GraphState(TypedDict):
    """
    Shared state that flows through every node in the LangGraph graph.

    Attributes:
        question     : the user's original question
        documents    : list of retrieved text chunks
        generation   : the LLM's generated answer
        is_relevant  : did retrieved docs pass the grading step?
        is_grounded  : is the answer grounded in the documents?
        retries      : how many times we have tried to regenerate
    """
    question    : str
    documents   : List[str]
    generation  : Optional[str]
    is_relevant : Optional[bool]
    is_grounded : Optional[bool]
    retries     : int