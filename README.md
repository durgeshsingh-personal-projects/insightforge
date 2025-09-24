InsightForge 🧠

An AI-powered Retrieval-Augmented Generation (RAG) assistant that answers domain-specific questions by retrieving relevant context from a knowledge base and combining it with LLM reasoning.

🚀 Features

Document ingestion & vector store (FAISS / Chroma) for semantic search.

RAG pipeline powered by LangChain + OpenAI models.

Evaluation framework to benchmark generated answers.

Streamlit app for interactive Q&A.

Extensible design for HR, enterprise, or domain-specific assistants.

📂 Project Structure
insightforge/
├── src/
│   ├── knowledge_base.py   # Vector store creation & loading
│   ├── rag_chain.py        # RAG pipeline (retrieval + LLM)
│   ├── evaluation.py       # Evaluation of QA predictions
│   ├── streamlit_app.py    # Frontend (Streamlit UI)
│   └── ...
├── data/                   # Source documents
├── .env                    # API keys (not committed)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

⚙️ Setup
1. Clone Repository
git clone https://github.com/yourusername/insightforge.git
cd insightforge

2. Create Virtual Environment
python3 -m venv .venv
source .venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

4. Configure API Keys

Create a .env file:

OPENAI_API_KEY=your_openai_api_key_here

🧩 Usage
1. Build Knowledge Base

Put your documents (.pdf, .txt, .md) in the data/ folder, then run:

python3 -m src.knowledge_base

2. Run Streamlit App
streamlit run src/streamlit_app.py


This launches the InsightForge UI in your browser.

3. Query the Assistant

Ask domain-specific questions and get context-aware answers.

📊 Evaluation

We use LangChain’s QA EvalChain + manual checks.

Run:

python3 -m src.evaluation


Expected output:

Predictions for sample queries

Side-by-side comparison with gold answers

✅ / ❌ correctness checks
