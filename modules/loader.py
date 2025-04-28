import os
import bs4
from langchain_community.document_loaders import (
    WebBaseLoader,
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader
)
import tempfile
import logging
import requests
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_documents(url: str):
    """
    Load documents from a web URL. Handles both HTML pages and direct PDF links.

    Args:
        url: The URL to load documents from.

    Returns:
        A list of LangChain Document objects, or an empty list if loading fails.
    """
    logger.info(f"Attempting to load documents from URL: {url}")
    docs = []
    is_pdf_url = False

    try:
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()

        if path.endswith('.pdf'):
            is_pdf_url = True
       

    except Exception as e:
        logger.warning(f"URL parsing/checking error for {url}: {e}. Assuming non-PDF.")

    if is_pdf_url:
        logger.info(f"Detected PDF URL: {url}. Downloading and parsing.")
        temp_file_path = None
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_file_path = temp_file.name
            logger.info(f"PDF from {url} saved temporarily to '{temp_file_path}'")

            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
            logger.info(f"Successfully loaded {len(docs)} pages from PDF URL: {url}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download PDF from {url}: {e}")
          
            raise RuntimeError(f"Network error downloading PDF from URL: {e}") from e
        except Exception as e:
            logger.error(f"Failed to process downloaded PDF from {url}: {e}")
            raise RuntimeError(f"Error processing PDF content from URL: {e}") from e
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.info(f"Removed temporary PDF file: {temp_file_path}")
                except Exception as e_clean:
                    logger.error(f"Error removing temporary PDF file {temp_file_path}: {e_clean}")

    else:
        logger.info(f"Processing URL as HTML page: {url}")
        try:
         
            loader = WebBaseLoader(web_paths=(url,))
            docs = loader.load()
            
            if not docs or all(not getattr(d, 'page_content', '').strip() for d in docs):
                 logger.warning(f"WebBaseLoader loaded {len(docs)} doc(s), but they seem empty for {url}. The page might require JavaScript or be blocked.")

            else:
                logger.info(f"Successfully loaded {len(docs)} document section(s) from HTML URL: {url}")

        except Exception as e:
            logger.error(f"Failed to load or parse HTML from {url}: {e}")
           
            docs = []

    return docs


def load_file(uploaded_file):
    """
    Load and process an uploaded file (PDF, TXT, DOCX).

    Args:
        uploaded_file: The file object uploaded via Streamlit's file_uploader.

    Returns:
        A list of LangChain Document objects.
    """
    docs = []
    temp_file_path = None
    file_name = uploaded_file.name
    logger.info(f"Attempting to load uploaded file: {file_name}")

    try:
        suffix = os.path.splitext(file_name)[1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        logger.info(f"Uploaded file '{file_name}' saved temporarily to '{temp_file_path}'")

        if suffix == ".pdf":
            loader = PyPDFLoader(temp_file_path)
        elif suffix == ".txt":
            loader = TextLoader(temp_file_path, encoding='utf-8') 
        elif suffix == ".docx":
            loader = UnstructuredWordDocumentLoader(temp_file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}. Please upload .pdf, .txt, or .docx")

        docs = loader.load()
        logger.info(f"Successfully loaded {len(docs)} document section(s) from file '{file_name}'")

    except Exception as e:
        logger.error(f"Error loading file '{file_name}': {e}")
      
        raise e
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Removed temporary uploaded file: {temp_file_path}")
            except Exception as e_clean:
                logger.error(f"Error removing temporary file {temp_file_path}: {e_clean}")

    return docs