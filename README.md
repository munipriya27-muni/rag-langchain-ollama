# rag-langchain-ollama
A local PDF-based RAG AI assistant built with LangChain, FAISS, Hugging Face embeddings, Ollama, and Llama 3.2.
# 🤖 RAG AI Assistant

A local **Retrieval-Augmented Generation (RAG)** application built with **Python, LangChain, FAISS, Hugging Face Embeddings, Ollama, and Llama 3.2 3B**.

The application allows users to upload a PDF document and ask questions about its content. Relevant information is retrieved from the document and provided to a locally running LLM to generate a context-aware answer.

## 🚀 Features

* 📄 Upload PDF documents
* ✂️ Split documents into smaller text chunks
* 🧠 Generate embeddings using `all-MiniLM-L6-v2`
* 🔎 Store and retrieve document embeddings using FAISS
* 🤖 Run Llama 3.2 3B locally using Ollama
* 💬 Ask questions about uploaded documents
* 📚 Display retrieved source content and page numbers
* ⚡ Cache embeddings, LLM, and vector store resources
* 🎨 Interactive Streamlit interface with a dark-themed UI

## 🏗️ RAG Architecture

```text
                PDF Document
                     │
                     ▼
              PDF Document Loader
                     │
                     ▼
              Text Chunking
                     │
                     ▼
            Hugging Face Embeddings
                     │
                     ▼
                FAISS Vector Store
                     │
                     ▼
                User Question
                     │
                     ▼
                 Retriever
                (Top 3 Chunks)
                     │
                     ▼
              Retrieved Context
                     │
                     ▼
             Ollama - Llama 3.2
                     │
                     ▼
              Generated Answer
```

## 🛠️ Technologies Used

| Technology                     | Purpose                   |
| ------------------------------ | ------------------------- |
| Python                         | Programming language      |
| Streamlit                      | Web application interface |
| LangChain                      | RAG application framework |
| PyPDFLoader                    | PDF document loading      |
| RecursiveCharacterTextSplitter | Text chunking             |
| Hugging Face                   | Text embeddings           |
| FAISS                          | Vector similarity search  |
| Ollama                         | Local LLM execution       |
| Llama 3.2 3B                   | Local language model      |

## 🔄 How It Works

### 1. Upload a PDF

The user uploads a PDF document through the Streamlit interface.

### 2. Load and Split the Document

The PDF is loaded using `PyPDFLoader` and divided into smaller chunks using `RecursiveCharacterTextSplitter`.

The current configuration uses:

```python
chunk_size=500
chunk_overlap=50
```

### 3. Generate Embeddings

Each chunk is converted into a vector representation using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

### 4. Create the Vector Store

The generated embeddings are stored in a **FAISS vector database**, allowing relevant chunks to be retrieved efficiently.

### 5. Retrieve Relevant Information

When the user asks a question, the system retrieves the **top 3 relevant chunks** from the document.

### 6. Generate the Answer

The retrieved content is passed as context to **Llama 3.2 3B**, running locally through Ollama.

The model is instructed to answer only using the retrieved document context and avoid making up information.

## 💻 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/rag-langchain-ollama.git
cd rag-langchain-ollama
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama

Install Ollama on your system and make sure it is running.

Then download the required model:

```bash
ollama pull llama3.2:3b
```

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📦 requirements.txt

The project uses the following main packages:

```text
streamlit
langchain-community
langchain-text-splitters
langchain-huggingface
langchain-ollama
faiss-cpu
sentence-transformers
pypdf
```

## 📸 Application

### Upload Document

Upload a PDF document through the application interface.

### Ask Questions

After the document is processed, enter a question related to its content.

### Retrieved Sources

The application displays the retrieved source chunks along with their corresponding page numbers.

## 🎯 Project Objective

This project was created as a **hands-on learning project to understand the practical implementation of Retrieval-Augmented Generation (RAG)**.

The main goal was to understand how an LLM can use information retrieved from an external document instead of relying only on its pre-trained knowledge.

## 📚 What I Learned

* How a basic RAG pipeline works
* Document loading and preprocessing
* Text chunking strategies
* Creating and storing embeddings
* Vector similarity search using FAISS
* Connecting LangChain with a local LLM
* Running open-source LLMs locally using Ollama
* Passing retrieved context to an LLM
* Building an interactive RAG application using Streamlit

## 🔮 Future Improvements

* Support multiple document formats
* Allow multiple PDF uploads
* Add conversation memory
* Improve retrieval using hybrid search
* Add streaming responses
* Add document citation and source highlighting
* Experiment with different open-source LLMs and embedding models

## 👨‍💻 Author

Munipriya

B.Tech | Computer Science & Engineering
Artificial Intelligence & Machine Learning

---

⭐ If you found this project useful, feel free to star the repository!
