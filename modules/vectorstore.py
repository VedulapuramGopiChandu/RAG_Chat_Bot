from langchain_community.vectorstores import FAISS

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

def create_vectorstore(documents: list[Document], embedding: Embeddings):
    """
    Creates an IN-MEMORY FAISS vector store from documents and an embedding model.

    Args:
        documents: The list of chunked Document objects.
        embedding: The embedding model instance (e.g., GoogleGenerativeAIEmbeddings).

    Returns:
        A FAISS vector store instance.
    """
    if not documents:
        logger.error("create_vectorstore (FAISS) received empty documents list.")
        raise ValueError("Cannot create vector store from empty documents.")

    logger.info(f"Creating FAISS in-memory vector store with {len(documents)} chunks...")
    try:
 
        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=embedding
        )
        logger.info("Successfully created FAISS vector store.")
        return vectorstore

    except ImportError as ie:
        
         logger.error(f"ImportError creating FAISS vector store: {ie}. Is 'faiss-cpu' or 'faiss-gpu' installed?", exc_info=True)
         raise RuntimeError("FAISS library not found. Please install it (`pip install faiss-cpu`) and add to requirements.txt.") from ie
    except Exception as e:
        
        logger.error(f"Failed to create FAISS vector store: {e}", exc_info=True)
        raise RuntimeError(f"Failed to create FAISS vector store: {e}") from e