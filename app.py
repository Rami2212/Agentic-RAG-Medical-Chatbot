import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import time
from datetime import datetime
from data_loader import init_data
from workflow import get_compiled_workflow, GraphState


def initialize_session_state():
    """Initialize Streamlit session state."""
    if "agentic_rag" not in st.session_state:
        st.session_state.agentic_rag = get_compiled_workflow()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "data_initialized" not in st.session_state:
        st.session_state.data_initialized = False


def init_data_on_startup():
    """Initialize data if not already done."""
    if not st.session_state.data_initialized:
        with st.spinner("Loading medical data and initializing ChromaDB..."):
            init_data()
            st.session_state.data_initialized = True
        st.success("✓ Data initialized successfully!")


def display_header():
    """Display application header."""
    st.set_page_config(
        page_title="Medical Chatbot",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🏥 Medical Chatbot - Agentic RAG")
        st.caption("Intelligent medical question answering with retrieval-augmented generation")
    with col2:
        st.metric("Data Status", "✓ Ready" if st.session_state.data_initialized else "Loading...")


def display_sidebar():
    """Display sidebar with information."""
    with st.sidebar:
        st.header("ℹ️ About")
        st.write(
            "This application uses an agentic RAG system to answer medical questions. "
            "It intelligently routes queries to different knowledge sources."
        )

        st.subheader("📚 Knowledge Sources")
        st.write("""
        - **Medical QnA**: General medical knowledge, symptoms, treatments
        - **Medical Devices**: Device manuals, specifications, usage
        - **Web Search**: Recent news and external data
        """)

        st.subheader("⚙️ Settings")
        if st.button("🔄 Clear Chat History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")


def display_chat_message(role: str, content: str, metadata: dict = None):
    """Display a chat message."""
    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.write(content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.write(content)
            if metadata:
                with st.expander("📊 Process Details"):
                    cols = st.columns(2)
                    with cols[0]:
                        st.write(f"**Source**: {metadata.get('source', 'N/A')}")
                        st.write(f"**Relevant**: {metadata.get('is_relevant', 'N/A')}")
                    with cols[1]:
                        st.write(f"**Router Decision**: {metadata.get('router_decision', 'N/A')}")


def create_process_display(status: str, message: str = ""):
    """Create a visual process indicator."""
    status_icons = {
        "routing": "🔀",
        "retrieving": "📚",
        "checking": "✓",
        "augmenting": "✨",
        "generating": "🤖",
        "complete": "✅",
        "error": "❌",
    }

    icon = status_icons.get(status, "⏳")
    return f"{icon} {message}"


def display_stream_process(state_history: list):
    """Display the streaming process in real-time."""
    process_container = st.container()

    with process_container:
        st.subheader("🔄 Processing Stream")

        # Create columns for process flow
        cols = st.columns(5)
        steps = [
            ("Routing", "🔀"),
            ("Retrieving", "📚"),
            ("Checking", "✓"),
            ("Augmenting", "✨"),
            ("Generating", "🤖")
        ]

        for i, (step_name, icon) in enumerate(steps):
            with cols[i]:
                st.write(f"{icon} {step_name}")

    return process_container


def run_query(query: str, progress_placeholder):
    """Run the agentic workflow with process visualization."""
    workflow = st.session_state.agentic_rag

    state_history = []
    current_node = None

    # Create initial state
    initial_state: GraphState = {
        "query": query,
        "context": "",
        "prompt": "",
        "response": "",
        "source": "",
        "router_decision": "",
        "is_relevant": "",
        "iteration_count": 0,
    }

    try:
        # Stream the workflow execution
        for output in workflow.stream(initial_state):
            # Get the node that just executed
            if output:
                node_name = list(output.keys())[0]
                state_data = output[node_name]

                # Update process display
                step_messages = {
                    "Router": create_process_display("routing", "Routing query..."),
                    "Retrieve_QnA": create_process_display("retrieving", "Retrieving from Medical QnA..."),
                    "Retrieve_Medical_Devices": create_process_display("retrieving", "Retrieving from Medical Devices..."),
                    "Web_Search": create_process_display("retrieving", "Searching the web..."),
                    "Check_Relevance": create_process_display("checking", "Checking relevance..."),
                    "Augment": create_process_display("augmenting", "Augmenting prompt..."),
                    "Generate": create_process_display("generating", "Generating response..."),
                }

                if node_name in step_messages:
                    with progress_placeholder:
                        st.info(step_messages[node_name])

                state_history.append((node_name, state_data.copy()))
                current_node = node_name

            # Small delay for visual effect
            time.sleep(0.1)

        # Get final state
        final_state = state_history[-1][1] if state_history else initial_state

        with progress_placeholder:
            st.success(create_process_display("complete", "Processing complete!"))

        return final_state, state_history

    except Exception as e:
        with progress_placeholder:
            st.error(create_process_display("error", f"Error: {str(e)}"))
        return None, []


def main():
    """Main application function."""
    initialize_session_state()
    display_header()
    display_sidebar()

    # Initialize data
    init_data_on_startup()

    # Display chat history
    st.subheader("💬 Chat")

    for message in st.session_state.chat_history:
        display_chat_message(message["role"], message["content"], message.get("metadata"))

    # User input
    if query := st.chat_input("Ask a medical question..."):
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": query
        })

        # Display user message
        display_chat_message("user", query)

        # Process query with progress display
        progress_placeholder = st.empty()

        with st.spinner("🔄 Processing your query..."):
            final_state, state_history = run_query(query, progress_placeholder)

        if final_state:
            # Display response
            response = final_state.get("response", "No response generated")

            metadata = {
                "source": final_state.get("source", "Unknown"),
                "is_relevant": final_state.get("is_relevant", "Unknown"),
                "router_decision": final_state.get("router_decision", "Unknown"),
            }

            # Add assistant message to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "metadata": metadata
            })

            # Display assistant message
            display_chat_message("assistant", response, metadata)

            # Display detailed process flow (in expander)
            with st.expander("📋 Detailed Process Flow"):
                for i, (node_name, state) in enumerate(state_history, 1):
                    st.write(f"**Step {i}: {node_name}**")
                    if state.get("context"):
                        st.write(f"Context: {state['context'][:200]}...")
                    if state.get("is_relevant"):
                        st.write(f"Relevant: {state['is_relevant']}")


if __name__ == "__main__":
    main()

