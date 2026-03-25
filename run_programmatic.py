from data_loader import init_data
from workflow import get_compiled_workflow


def main():
    """Run the medical chatbot workflow."""

    # Initialize data
    print("Initializing data and ChromaDB...")
    collection1, collection2 = init_data()
    print(f"✓ QnA Collection: {collection1.count()} documents")
    print(f"✓ Device Collection: {collection2.count()} documents")

    # Get the compiled workflow
    print("\nCompiling workflow...")
    agentic_rag = get_compiled_workflow()
    print("✓ Workflow compiled successfully!")

    # Example query
    query = "What are the devices relevant to diabetes management?"

    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}\n")

    # Run the workflow
    initial_state = {
        "query": query,
        "context": "",
        "prompt": "",
        "response": "",
        "source": "",
        "router_decision": "",
        "is_relevant": "",
        "iteration_count": 0,
    }

    # Stream the execution
    for output in agentic_rag.stream(initial_state):
        if output:
            node_name = list(output.keys())[0]
            state = output[node_name]
            print(f"\n▶ Node: {node_name}")

            if state.get("router_decision"):
                print(f"  → Router Decision: {state['router_decision']}")
            if state.get("source"):
                print(f"  → Source: {state['source']}")
            if state.get("is_relevant"):
                print(f"  → Relevant: {state['is_relevant']}")
            if state.get("context"):
                print(f"  → Context: {state['context'][:100]}...")
            if state.get("response"):
                print(f"  → Response: {state['response']}")

    print(f"\n{'='*60}")
    print("✓ Workflow execution completed!")


if __name__ == "__main__":
    main()

