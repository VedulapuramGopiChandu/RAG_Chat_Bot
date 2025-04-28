from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStoreRetriever
import logging

logger = logging.getLogger(__name__)

def build_rag_chain(retriever: VectorStoreRetriever, model: BaseChatModel):
    """
    Builds a conversational RAG chain that uses chat history.

    Args:
        retriever: The vector store retriever instance.
        model: The chat model instance (e.g., ChatGoogleGenerativeAI).

    Returns:
        The runnable conversational RAG chain.
    """
    logger.info("Building the CONVERSATIONAL RAG chain...")

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"), 
            ("human", "{input}"), 
        ]
    )
    try:
        history_aware_retriever_chain = create_history_aware_retriever(
            model, retriever, contextualize_q_prompt
        )
        logger.info("Created history-aware retriever chain.")
    except Exception as e:
         logger.error(f"Failed to create history-aware retriever: {e}", exc_info=True)
         raise RuntimeError(f"Could not build the history-aware retriever component: {e}") from e

    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer based on the context, just say that you don't know. "
        "Keep your answers concise and based *only* on the provided context. "
        "Do not add information that is not present in the context."
        "\n\n"
        "Context:\n{context}" 
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"), 
            ("human", "{input}"), 
        ]
    )
    try:
       
        question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
        logger.info("Created question answering chain (stuff documents).")
    except Exception as e:
         logger.error(f"Failed to create question answer chain: {e}", exc_info=True)
         raise RuntimeError(f"Could not build the question answering component: {e}") from e

    try:
        rag_chain = create_retrieval_chain(history_aware_retriever_chain, question_answer_chain)
        logger.info("Created final conversational retrieval chain.")
        return rag_chain
    except Exception as e:
         logger.error(f"Failed to create final retrieval chain: {e}", exc_info=True)
         raise RuntimeError(f"Could not build the final conversational RAG chain: {e}") from e