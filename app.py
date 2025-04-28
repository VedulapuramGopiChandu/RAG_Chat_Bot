import streamlit as st
from dotenv import load_dotenv
import warnings
import logging
import os

from langchain_core.messages import HumanMessage, AIMessage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


load_dotenv()
logger.info(".env file loaded (if exists).")


warnings.filterwarnings('ignore')


EMBEDDING_MODEL_NAME = "models/embedding-001"
CHAT_MODEL_NAME = "gemini-1.5-pro"


try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
  
    from modules.loader import load_documents, load_file
    from modules.splitter import split_documents
    from modules.vectorstore import create_vectorstore 
    from modules.rag_chain import build_rag_chain
except ImportError as e:
   
    logger.error(f"Error importing required libraries: {e}")
    st.error(f"Fatal Error: Required library not found. Please install dependencies from requirements.txt. Details: {e}")
    st.stop()

try:
    embeddings_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    chat_model = ChatGoogleGenerativeAI(
        model=CHAT_MODEL_NAME,
        convert_system_message_to_human=True,
        temperature=0.5,
    )
    logger.info("Successfully initialized Google AI models.")
except Exception as e:
  
    logger.error(f"Error initializing Google AI models: {e}", exc_info=True)
    st.error(f"🚨 Error initializing Google AI models: {e}")
    st.info("Please check your GOOGLE_API_KEY, model names, and internet connection.")
    st.stop()


st.set_page_config(
    page_title="Smart RAG App: Web & File Chat",
    page_icon="🤖",
    layout="wide"
)
st.title(" Smart RAG App: Chat with Webpages and Files")
st.markdown("Load content from a public URL or upload a file (.pdf, .txt, .docx), then ask questions!")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    logger.debug("Session state 'vectorstore' initialized to None.")
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
    logger.debug("Session state 'rag_chain' initialized to None.")
if "data_source_info" not in st.session_state:
    st.session_state.data_source_info = None
    logger.debug("Session state 'data_source_info' initialized to None.")

if "messages" not in st.session_state:
    st.session_state.messages = [] 
    logger.debug("Session state 'messages' initialized to empty list.")


def process_and_store_data(docs: list, source_description: str):
    """Splits, vectorizes, and builds the RAG chain, updating session state."""
    try:
        if not docs:
          
            return False

      
        splits = split_documents(docs)
        if not splits:
           
            return False
        st.write(f" Split into {len(splits)} chunks.")
        logger.info(f"Split {len(docs)} docs into {len(splits)} chunks for source: {source_description}")

        vectorstore = create_vectorstore(splits, embeddings_model)
        st.write("Created vector store.")
        logger.info(f"Created vector store for source: {source_description}")

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 5})
        rag_chain = build_rag_chain(retriever, chat_model) # Calls the modified function
        st.write("Built Conversational RAG chain.")
        logger.info(f"Built Conversational RAG chain for source: {source_description}")

        st.session_state.vectorstore = vectorstore
        st.session_state.rag_chain = rag_chain
        st.session_state.data_source_info = source_description

        st.session_state.messages = []
        logger.info("Cleared chat history for new data source.")

        st.success(f"Successfully processed '{source_description}'! You can now ask questions below.")
        logger.info(f"Successfully updated session state for source: {source_description}")
        return True

    except Exception as e:

        st.session_state.vectorstore = None
        st.session_state.rag_chain = None
        st.session_state.data_source_info = None
        st.session_state.messages = [] 
        return False


tab1, tab2 = st.tabs(["🌐 Load from URL", "📄 Upload a File"])

with tab1:
    st.subheader("Load Content from a Webpage URL")
    url_input = st.text_input(
        "Enter a public URL (HTML or PDF):",
        placeholder="https://example.com or https://example.com/document.pdf",
        key="url_input_widget"
        )

    if st.button("Load URL Data", key="load_url_button"):
        if url_input:
            logger.info(f"User initiated URL load: {url_input}")
            with st.spinner(f'Processing URL: {url_input}... Please wait.'):
                try:
                    loaded_docs = load_documents(url_input)
                    st.write(f"Attempted loading content from: {url_input}")
                    if loaded_docs:
                         st.write(f" Loaded {len(loaded_docs)} document section(s).")
                         process_and_store_data(loaded_docs, f"URL: {url_input}")
                    else:
                         process_and_store_data(loaded_docs, f"URL: {url_input}")

                except Exception as e:
                    st.error(f"Error loading or processing URL: {str(e)}")
                    logger.error(f"Exception during URL processing ({url_input}): {e}", exc_info=True)
                    st.session_state.vectorstore = None
                    st.session_state.rag_chain = None
                    st.session_state.data_source_info = None
                    st.session_state.messages = []
        else:
            st.warning("Please enter a URL.")

with tab2:
    st.subheader("Load Content from a File")
    uploaded_file = st.file_uploader(
        "Upload a file (.pdf, .txt, .docx):",
        type=["pdf", "txt", "docx"],
        key="file_uploader_widget"
    )

    if st.button("Load File Data", key="load_file_button"):
        if uploaded_file is not None:
            file_name = uploaded_file.name
            logger.info(f"User initiated file upload: {file_name}")
            with st.spinner(f'Processing file: "{file_name}"... Please wait.'):
                try:
                    loaded_docs = load_file(uploaded_file)
                    st.write(f"Attempted loading content from file: {file_name}")
                    if loaded_docs:
                        st.write(f"Loaded {len(loaded_docs)} document section(s).")
                        process_and_store_data(loaded_docs, f"File: {file_name}")
                    else:
                        process_and_store_data(loaded_docs, f"File: {file_name}")

                except Exception as e:
                    st.error(f"Error loading or processing file: {str(e)}")
                    logger.error(f"Exception during file processing ({file_name}): {e}", exc_info=True)
                    st.session_state.vectorstore = None
                    st.session_state.rag_chain = None
                    st.session_state.data_source_info = None
                    st.session_state.messages = [] 
            st.warning("Please upload a file.")


st.divider()

if st.session_state.rag_chain:
    st.header(f"Ask Questions about '{st.session_state.data_source_info}'")

    for message in st.session_state.messages:
        with st.chat_message(message.type): 
            st.markdown(message.content)
 
    if user_query := st.chat_input("Ask your question here...", key="chat_input_widget"):
        logger.info(f"User query: {user_query}")

        with st.chat_message("human"):
            st.markdown(user_query)
        st.session_state.messages.append(HumanMessage(content=user_query))


        chain_input = {
            "input": user_query,
            "chat_history": st.session_state.messages[:-1]
        }

        with st.spinner("Thinking..."):
            try:
            
                response = st.session_state.rag_chain.invoke(chain_input)
                answer = response.get("answer", "Sorry, I couldn't generate an answer based on the context.")
                logger.info(f"Conversational RAG chain response generated.")

                with st.chat_message("ai"):
                    st.markdown(answer)
                st.session_state.messages.append(AIMessage(content=answer))


            except Exception as e:
                error_message = f"An error occurred while getting the answer: {str(e)}"
                st.error(error_message)
                logger.error(f"Error during conversational RAG chain invocation: {e}", exc_info=True)
               
elif st.session_state.data_source_info:
     st.warning(f"Processing failed for '{st.session_state.data_source_info}'. Please try loading again or check the logs.")
     logger.warning(f"RAG chain not available, but data source info exists: {st.session_state.data_source_info}. Processing likely failed.")
     st.session_state.vectorstore = None
     st.session_state.rag_chain = None
     st.session_state.data_source_info = None
     st.session_state.messages = [] 

else:
    st.info("Load data using the tabs above to begin chatting.")
    logger.info("App loaded. Waiting for user to load data.")