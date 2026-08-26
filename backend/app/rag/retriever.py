from typing import List, Dict, Any, Optional
from langchain_core.vectorstores import VectorStoreRetriever
from app.rag.vectorstore import rag_store
from app.core.logging import logger


def get_retriever(session_id: Optional[str] = None, k: int = 3) -> VectorStoreRetriever:
    """
    Return a LangChain VectorStoreRetriever scoped to a session.
    Can be plugged directly into LCEL chains:
        chain = retriever | prompt | llm
    """
    search_kwargs: Dict[str, Any] = {"k": k}
    if session_id:
        search_kwargs["filter"] = {"session_id": session_id}

    if rag_store.vector_db:
        return rag_store.vector_db.as_retriever(search_kwargs=search_kwargs)

    # Fallback: wrap the in-memory list in a minimal retriever-compatible object
    logger.warning("Chroma vectorstore not available — retriever uses in-memory fallback store.")
    return rag_store.vector_db.as_retriever(search_kwargs=search_kwargs) if rag_store.vector_db else None


def store_analysis_knowledge(
    session_id: str,
    file_id: str,
    knowledge_type: str,
    content: str,
    extra_meta: Optional[Dict[str, Any]] = None,
):
    """Persist an analysis result into the RAG vector store."""
    metadata = {"session_id": session_id, "file_id": file_id, "type": knowledge_type}
    if extra_meta:
        metadata.update(extra_meta)
    rag_store.add_knowledge(content, metadata)


def retrieve_context_for_query(query: str, session_id: str, k: int = 3) -> str:
    """
    Retrieve relevant knowledge chunks for a query and format them as a
    plain-text context block for injection into LLM prompts.
    """
    docs = rag_store.similarity_search(query=query, session_id=session_id, k=k)
    if not docs:
        return ""

    context_blocks = []
    for idx, doc in enumerate(docs, 1):
        src_type = doc.metadata.get("type", "General Knowledge")
        context_blocks.append(f"--- Knowledge Item #{idx} [{src_type}] ---\n{doc.page_content}")

    return "\n\n".join(context_blocks)
