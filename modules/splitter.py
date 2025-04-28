from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

def split_documents(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 150) -> list[Document]:
    """
    Splits a list of LangChain Documents into smaller chunks.

    Args:
        documents: The list of Document objects to split.
        chunk_size: The maximum size of each chunk (in characters).
        chunk_overlap: The number of characters to overlap between chunks.

    Returns:
        A list of chunked Document objects.
    """
    if not documents:
        logger.warning("split_documents received an empty list of documents.")
        return []

    logger.info(f"Splitting {len(documents)} document(s) into chunks (size={chunk_size}, overlap={chunk_overlap})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len, 
        add_start_index=True, 
    )
    try:
        splits = splitter.split_documents(documents)
        logger.info(f"Successfully split documents into {len(splits)} chunks.")
        return splits
    except Exception as e:
        logger.error(f"Error during document splitting: {e}")
       
        raise RuntimeError(f"Failed to split documents: {e}") from e