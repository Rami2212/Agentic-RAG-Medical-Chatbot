# Agentic-RAG-Medical-Chatbot

**A production-ready Agentic RAG (Retrieval-Augmented Generation) pipeline** using LangGraph and ChromaDB with modular agents, intelligent retrieval, and end-to-end orchestration.

Build powerful medical question-answering systems with real-time process visualization, intelligent routing, and comprehensive document retrieval.

---

## ⚡ Quick Start

### Prerequisites
- Python 3.12
- OpenAI API key

### Installation & Run

```bash
# 1. Clone and navigate
git clone <repository-url>
cd Agentic-RAG-Medical-Chatbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API key - Create .env file
OPENAI_API_KEY=your_openai_api_key_here
LLM_MODEL=model

# 4. Run the Streamlit app
python -m streamlit run app.py
```

**Access at**: http://localhost:8501

---

## 🎯 Key Features

### 🎨 Beautiful Streamlit Interface
- **Intuitive Chat Interface**: Familiar messaging layout
- **Real-time Visualization**: Watch the AI decision-making process live
- **Chat History**: Persistent conversation memory
- **Process Metadata**: See which sources were used and confidence scores

### 🤖 Intelligent AI Architecture
- **Agentic Routing**: Automatically selects the best knowledge source
- **Three Knowledge Sources**:
  - 📚 Medical Q&A Collection (500 documents)
  - 🔧 Medical Device Manuals (500 documents)
  - 🌐 Web Search Fallback
- **Relevance Checking**: Validates that retrieved context answers the query
- **RAG Pipeline**: Augments queries with relevant context for accurate responses

### 🏗️ Modular Architecture
- **6 Clean Python Modules**: Each with single responsibility
- **Easy to Extend**: Add new data sources or processing steps
- **Well-Documented**: Comprehensive inline documentation and guides
- **Production-Ready**: Error handling, type hints, configuration management

### ⚡ Performance & Scalability
- **ChromaDB Vector Database**: Efficient semantic search across 1000+ documents
- **Session-Based State Management**: Handles multiple concurrent users
- **Configurable Parameters**: Fine-tune retrieval and generation settings
- **Streaming Responses**: Real-time feedback during processing

---

## 📋 Project Structure

```
Agentic-RAG-Medical-Chatbot/
│
├── 💻 Core Modules
│   ├── config.py                
│   ├── llm_service.py          
│   ├── data_loader.py          
│   ├── retrieval.py            
│   ├── workflow.py            
│   └── app.py                 
│
├── 📦 Configuration & Utilities
│   ├── requirements.txt      
│   ├── run_programmatic.py     
│   ├── .env     
│   ├── README.md              
│   └── LICENSE                
│
└── 📊 Data
    ├── data/
    │   ├── medical_qna_dataset.csv       
    │   └── medical_device_manuals_dataset.csv  
    └── chroma_db/             
```

## 🤖 How It Works

### Workflow Pipeline

```
User Input
    ↓
┌─────────────────────────────────────┐
│ 🔀 Router Node                      │
│ → Analyzes query type               │
│ → Selects information source        │
└─────────────────────────────────────┘
    ↓
    ├──→ 📚 Retrieve QnA
    ├──→ 🔧 Retrieve Device Manuals
    └──→ 🌐 Web Search
    ↓
┌─────────────────────────────────────┐
│ ✓ Check Relevance Node              │
│ → Validates context relevance       │
│ → Ensures answer quality            │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ ✨ Augment Node                     │
│ → Builds RAG prompt                 │
│ → Incorporates retrieved context    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 🤖 Generate Node                    │
│ → Calls OpenAI API                  │
│ → Creates final response            │
└─────────────────────────────────────┘
    ↓
Response with Metadata
```

### Key Concepts

**Retrieval-Augmented Generation (RAG)**
- Combines document retrieval with language models
- Grounds responses in actual data sources
- Reduces hallucinations and improves accuracy

**Agentic Routing**
- Intelligent decision-making at each step
- Routes queries to most relevant data sources
- Adapts based on relevance checks

**LangGraph Workflow**
- State-based execution model
- Conditional routing and branching
- Seamless LLM integration

---

## 📦 Dependencies

Core dependencies (see `requirements.txt` for versions):

- **streamlit** - Web interface
- **langchain** - LLM framework
- **langgraph** - Workflow orchestration
- **chromadb** - Vector database
- **openai** - OpenAI API
- **pandas** - Data processing
- **python-dotenv** - Environment variables

---

## 📄 License

MIT License - see LICENSE file for details