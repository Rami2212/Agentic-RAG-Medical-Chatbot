from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph, START, END
from retrieval import (
    router,
    retrieve_context_qna,
    retrieve_context_medical_devices,
    web_search,
    check_relevance,
    build_prompt,
    call_llm,
)
from config import MAX_ITERATIONS


class GraphState(TypedDict):
    """State for the agentic workflow."""
    query: str
    context: str
    prompt: str
    response: str
    source: str
    router_decision: str
    is_relevant: str
    iteration_count: int


def route_on_retrieval(state: Dict[str, Any]) -> str:
    """Route based on router decision."""
    decision = state.get("router_decision", "Retrieve_QnA").strip()

    mapping = {
        "Retrieve_QnA": "Retrieve_QnA",
        "Retrieve_Medical_Devices": "Retrieve_Medical_Devices",
        "Web_Search": "Web_Search",
    }

    return mapping.get(decision, "Retrieve_QnA")


def route_on_relevance(state: Dict[str, Any]) -> str:
    """Route based on relevance check."""
    iteration_count = state.get("iteration_count", 0)
    iteration_count += 1
    state["iteration_count"] = iteration_count

    if iteration_count >= MAX_ITERATIONS:
        state["is_relevant"] = "Yes"

    is_relevant = state.get("is_relevant", "Yes").strip()

    if is_relevant.lower() == "yes":
        return "Augment"
    else:
        return "Web_Search"


def build_workflow():
    """Build and compile the workflow graph."""
    workflow = StateGraph(GraphState)

    workflow.add_node("Router", router)
    workflow.add_node("Retrieve_QnA", retrieve_context_qna)
    workflow.add_node("Retrieve_Medical_Devices", retrieve_context_medical_devices)
    workflow.add_node("Web_Search", web_search)
    workflow.add_node("Check_Relevance", check_relevance)
    workflow.add_node("Augment", build_prompt)
    workflow.add_node("Generate", call_llm)

    workflow.add_edge(START, "Router")

    workflow.add_conditional_edges(
        "Router",
        route_on_retrieval,
    )

    workflow.add_edge("Retrieve_QnA", "Check_Relevance")
    workflow.add_edge("Retrieve_Medical_Devices", "Check_Relevance")
    workflow.add_edge("Web_Search", "Check_Relevance")

    workflow.add_conditional_edges(
        "Check_Relevance",
        route_on_relevance,
    )

    workflow.add_edge("Augment", "Generate")
    workflow.add_edge("Generate", END)

    return workflow.compile()


def get_compiled_workflow():
    """Get the compiled workflow."""
    return build_workflow()

