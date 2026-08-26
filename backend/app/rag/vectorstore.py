import os
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Chroma
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document
from app.rag.embeddings import get_embeddings
from app.core.config import settings
from app.core.logging import logger

class RAGVectorStore:
    def __init__(self):
        self.embedding_func = get_embeddings()
        self.persist_directory = os.path.join(settings.BASE_DIR, "chroma_db")
        os.makedirs(self.persist_directory, exist_ok=True)
        self.fallback_docs = []
        self.vector_db = None

        try:
            self.vector_db = Chroma(
                collection_name="analytica_knowledge",
                embedding_function=self.embedding_func,
                persist_directory=self.persist_directory
            )
            logger.info("Initialized Chroma Vector Database successfully.")
        except Exception as e:
            logger.warning(f"Could not initialize Chroma vectorstore directly: {e}. Using in-memory fallback store.")

    def add_knowledge(self, content: str, metadata: Dict[str, Any]):
        # Ensure metadata values are str, int, float, or bool for Chroma compliance
        clean_metadata = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                clean_metadata[k] = v
            elif v is not None:
                clean_metadata[k] = str(v)

        doc = Document(page_content=content, metadata=clean_metadata)
        if self.vector_db:
            try:
                self.vector_db.add_documents([doc])
                return
            except Exception as e:
                logger.warning(f"Error adding document to vector database: {e}. Appending to in-memory fallback store.")
        
        self.fallback_docs.append(doc)

    def similarity_search(self, query: str, session_id: Optional[str] = None, k: int = 3) -> List[Document]:
        if self.vector_db:
            try:
                filter_dict = {"session_id": session_id} if session_id else None
                res = self.vector_db.similarity_search(query, k=k, filter=filter_dict)
                if res:
                    return res
            except Exception as e:
                logger.warning(f"Similarity search failed on vector store: {e}")

        # Fallback keyword matching search
        results = []
        q_lower = query.lower()
        for doc in self.fallback_docs:
            if session_id and doc.metadata.get("session_id") != session_id:
                continue
            if any(term in doc.page_content.lower() for term in q_lower.split()):
                results.append(doc)
            if len(results) >= k:
                break
        return results if results else self.fallback_docs[:k]

rag_store = RAGVectorStore()
