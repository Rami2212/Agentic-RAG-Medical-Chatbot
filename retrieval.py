from typing import Dict, Any
from langchain_tavily import TavilySearch
from llm_service import get_llm_response
from data_loader import get_collections
from config import TOP_K_RESULTS, TAVILY_TOPIC, TAVILY_MAX_RESULTS


def retrieve_context_qna(state: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve top documents from chromaDB collection (medical qna) based on the query."""
    collection1, _ = get_collections()

    query = state["query"]
    result = collection1.query(query_texts=[query], n_results=TOP_K_RESULTS)
    context = "\n".join(result["documents"][0])
    state["context"] = context
    state["source"] = "Medical QnA Collection"

    return state


def retrieve_context_medical_devices(state: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve top documents from chromaDB collection (medical device manuals) based on the query."""
    _, collection2 = get_collections()

    query = state["query"]
    result = collection2.query(query_texts=[query], n_results=TOP_K_RESULTS)
    context = "\n".join(result["documents"][0])
    state["context"] = context
    state["source"] = "Medical Device Manuals Collection"

    return state


def web_search(state: Dict[str, Any]) -> Dict[str, Any]:
    """Perform web search using the tavily search API."""
    tavily_search = TavilySearch(topic=TAVILY_TOPIC, max_results=TAVILY_MAX_RESULTS)
    query = state["query"]
    result = tavily_search.invoke({"query": query})
    state["context"] = result["results"][0]["content"] if result["results"] else "No results found"
    state["source"] = "Web Search"

    return state


def router(state: Dict[str, Any]) -> Dict[str, Any]:
    """Agentic router: decides which retrieval method to use."""
    decision_prompt = f"""
You are a routing agent. Based on the user query, decide where to look for information.

Options:
- Retrieve_QnA: If it's about general medical knowledge, symptoms, or treatments.
- Retrieve_Medical_Devices: If it's about medical devices, manuals, or instructions.
- Web_Search: If it's about recent news, brand names, or external data.

Query: "{state["query"]}"

Respond only with one of: "Retrieve_QnA", "Retrieve_Medical_Devices", "Web_Search"
"""
    router_decision = get_llm_response(decision_prompt).strip()
    state["router_decision"] = router_decision

    return state


def check_relevance(state: Dict[str, Any]) -> Dict[str, Any]:
    """Determine whether the retrieved context is relevant to the user query or not."""
    query = state["query"]
    context = state["context"]

    relevance_prompt = f"""
Check the context below to see if it is relevant to the user query.
####
Context: {context}
####
User Query: {query}

Options:
- Yes: if the context is relevant.
- No: if the context is not relevant.

Please answer with only "Yes" or "No".
"""
    relevance_decision_value = get_llm_response(relevance_prompt).strip()
    state["is_relevant"] = relevance_decision_value

    return state


def build_prompt(state: Dict[str, Any]) -> Dict[str, Any]:
    """Construct the RAG-style prompt."""
    query = state["query"]
    context = state["context"]

    from config import ANSWER_LENGTH_LIMIT

    prompt = f"""
Answer the following question using the context below.
Context:
{context}
Question: {query}
Please limit your answer to {ANSWER_LENGTH_LIMIT} words.
"""

    state["prompt"] = prompt
    return state


def call_llm(state: Dict[str, Any]) -> Dict[str, Any]:
    """Call the LLM to generate response."""
    prompt = state["prompt"]
    answer = get_llm_response(prompt)
    state["response"] = answer
    return state

