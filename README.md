# RAG_Chat_Bot
# Smart RAG App: Chat with Webpages and Files

### A simple chatbot that answers questions based on content from URLs or uploaded documents.

This app lets you "chat" with the information contained within a specific webpage or a document you upload (like PDFs, TXT files, or Word documents). Instead of just searching for keywords, it uses a Large Language Model (LLM) to try and *understand* the text and answer your questions based *only* on the information provided in that source.

This technique is often called **Retrieval-Augmented Generation (RAG)**. It "retrieves" relevant parts of the document and then "generates" an answer based on them.

##  Features

*   Load text content directly from a public webpage URL (handles regular HTML pages and direct links to PDF files).
*   Upload your own documents (.pdf, .txt, .docx).
*   Ask questions about the loaded content.
*   Engage in a basic conversation – the app remembers previous questions and answers in the current session to understand follow-up questions (like "tell me more about it").
*   Simple and interactive web interface built with Streamlit.

##  Tech Stack

This project uses several helpful Python libraries:

*   **Python:** The core programming language.
*   **Streamlit:** A fantastic library for quickly building interactive web apps like this one.
*   **LangChain:** A powerful framework that makes it easier to connect different components needed for building LLM applications (like loading documents, splitting text, managing prompts, and interacting with models).
*   **Google Generative AI:** Provides:
    *   The **Embedding Model** (`models/embedding-001`): Turns text chunks into numerical representations (vectors) so the app can understand their meaning and find similar chunks.
    *   The **Chat Model** (`gemini-1.5-pro`): The "brain" that reads the relevant text chunks and your question (plus conversation history) to generate a natural language answer.
*   **FAISS (Facebook AI Similarity Search):** A library used here as an efficient in-memory "vector store". It holds the numerical representations of the text chunks and allows the app to quickly find the chunks most relevant to your question.
*   **Other Libraries:** `python-dotenv` (for API keys), `pypdf`, `unstructured`, `python-docx` (for loading different file types), `requests`, `beautifulsoup4`, `lxml` (for loading web content).

## 🤔 How It Works (Simplified RAG Flow)

1.  **Load Data:** You provide a URL or upload a file. The app extracts the text content using LangChain's document loaders.
2.  **Split:** The extracted text is broken down into smaller, manageable chunks using a text splitter. This helps the model focus on relevant pieces.
3.  **Embed:** Each text chunk is converted into a list of numbers (an "embedding" or "vector") using the Google Embedding Model. These numbers represent the semantic meaning of the chunk.
4.  **Store:** These numerical embeddings (and the original text chunks) are stored in the FAISS vector store, which lives in your computer's memory while the app is running. FAISS is optimized for finding vectors that are "similar" to each other quickly.
5.  **Chat & Retrieve:**
    *   You ask a question in the chat interface.
    *   The app considers your current question and the previous conversation history. It might ask the LLM to rephrase your question into a better standalone search query.
    *   This query is converted into an embedding.
    *   FAISS searches the vector store to find the text chunks whose embeddings are numerically closest (most similar in meaning) to your question's embedding.
6.  **Generate:**
    *   The app takes your original question, the conversation history, and the relevant text chunks retrieved from FAISS.
    *   It sends all this information in a structured "prompt" to the Google Gemini chat model.
    *   The Gemini model reads the prompt and generates an answer based *only* on the provided text chunks and conversation context.
    *   The answer is displayed back to you in the chat.

## 💡 Possible Future Improvements

*   Add support for more document types (e.g., `.csv`, `.json`).
*   Improve handling of very large documents (potential memory issues).
*   Allow users to select different embedding or chat models.
*   Add more robust error handling and user feedback.
*   Implement document persistence (saving/loading FAISS index) if needed between runs (currently it's purely in-memory).
