# src/rag_chain.py

from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI

# Prompt template for summarization & recommendations
SUMMARY_PROMPT = PromptTemplate(
    input_variables=["context", "stats", "question"],
    template=(
        "You are InsightForge, an AI business analyst. Use the CONTEXT below (rows from the sales dataset) "
        "and the STATS (numeric aggregates) to answer the QUESTION. Provide:\n"
        "1) A concise one-paragraph summary of trends.\n"
        "2) Key metrics (3 bullets) pulled explicitly from STATS.\n"
        "3) 3 actionable recommendations for business stakeholders.\n\n"
        "CONTEXT:\n{context}\n\nSTATS:{stats}\n\nQUESTION:{question}\n\n"
        "Be explicit about assumptions and list any missing data that would help give better recommendations."
    )
)


def build_rag_chain(vectorstore):
    """
    Build a RetrievalQA chain using OpenAI LLM.
    """
    llm = OpenAI(temperature=0)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 'stuff' is fine for small datasets
        retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
        return_source_documents=True
    )
    return chain


def run_rag_answer(chain, query):
    """
    Run the RAG chain with a query.
    Returns a tuple: (AI answer string, list of source documents)
    """
    output = chain.invoke({"query": query})  # .invoke replaces deprecated .run()
    return output["result"], output["source_documents"]
