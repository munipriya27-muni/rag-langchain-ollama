import os
import tempfile

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: #0f1117;
    }

    /* Remove top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Header */
    .hero {
        padding: 30px;
        border-radius: 20px;
        background: linear-gradient(
            135deg,
            #171a24,
            #202637
        );
        border: 1px solid #303747;
        margin-bottom: 25px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        color: #aeb6c5;
        font-size: 17px;
    }

    /* Cards */
    .card {
        background: #171a24;
        border: 1px solid #2b3140;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 18px;
    }

    /* Status */
    .status {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        background: #17291f;
        border: 1px solid #285c3a;
        color: #6ee7a0;
        font-size: 13px;
    }

    /* Answer box */
    .answer-box {
        background: #171a24;
        border: 1px solid #303747;
        border-radius: 16px;
        padding: 25px;
        margin-top: 15px;
        line-height: 1.7;
        font-size: 16px;
    }

    /* Source */
    .source-box {
        background: #13161e;
        border-left: 3px solid #6c63ff;
        border-radius: 8px;
        padding: 15px;
        margin-top: 10px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #12151d;
        border-right: 1px solid #292f3b;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        padding: 10px;
        font-weight: 600;
    }

    /* Text input */
    div[data-baseweb="input"] {
        border-radius: 10px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🤖 RAG Assistant")

    st.markdown("---")

    st.markdown("### System")

    st.markdown(
        '<span class="status">● Ollama Connected</span>',
        unsafe_allow_html=True
    )

    st.markdown("")

    st.write("🧠 **LLM**")
    st.caption("Llama 3.2 3B")

    st.write("🔎 **Retriever**")
    st.caption("FAISS")

    st.write("📐 **Embeddings**")
    st.caption("all-MiniLM-L6-v2")

    st.markdown("---")

    st.markdown("### About")

    st.caption(
        "This application uses Retrieval-Augmented "
        "Generation (RAG) to answer questions from "
        "your document."
    )

    st.markdown("---")

    st.caption("Built with Python • LangChain • FAISS • Ollama")


# ============================================================
# HERO SECTION
# ============================================================

st.markdown("""
<div class="hero">
<div class="hero-title">📚 RAG AI Assistant</div>
<div class="hero-subtitle">Ask questions about your documents using Retrieval-Augmented Generation.</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE INIT
# ============================================================

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "documents" not in st.session_state:
    st.session_state.documents = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None


# ============================================================
# CACHED RESOURCES (loaded once, reused across reruns)
# ============================================================

@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )


@st.cache_resource(show_spinner=False)
def build_vectorstore(file_bytes, file_name):
    """
    Save the uploaded PDF to a temp file, load + split it,
    then build a FAISS vector store from the chunks.
    Cached on (file_bytes, file_name) so re-uploading the
    same file doesn't rebuild everything.
    """

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = text_splitter.split_documents(documents)

        embeddings = load_embeddings()

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

    finally:
        os.remove(tmp_path)

    return documents, chunks, vectorstore


# ============================================================
# UPLOAD DOCUMENT
# ============================================================

st.markdown("### 📤 Upload your document")

uploaded_file = st.file_uploader(
    "",
    type=["pdf"],
    label_visibility="collapsed"
)

if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()

    # Only (re)build the knowledge base if a new file was uploaded
    if st.session_state.file_name != uploaded_file.name:

        with st.spinner("📚 Reading document, chunking text, and building vector database..."):

            documents, chunks, vectorstore = build_vectorstore(
                file_bytes,
                uploaded_file.name
            )

        st.session_state.documents = documents
        st.session_state.chunks = chunks
        st.session_state.vectorstore = vectorstore
        st.session_state.file_name = uploaded_file.name

    st.success(f"✅ Loaded: {uploaded_file.name}")

else:
    st.info("Upload a PDF above to get started.")


llm = load_llm()


# ============================================================
# DOCUMENT INFO
# ============================================================

if st.session_state.vectorstore is not None:

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "📄 Pages",
            len(st.session_state.documents)
        )

    with col2:
        st.metric(
            "🧩 Chunks",
            len(st.session_state.chunks)
        )

    with col3:
        st.metric(
            "🔎 Retrieved",
            "Top 3"
        )

    st.markdown("<br>", unsafe_allow_html=True)


    # ========================================================
    # QUESTION
    # ========================================================

    st.markdown("### 💬 Ask your document")

    question = st.text_input(
        "",
        placeholder="Example: What is machine learning?",
        label_visibility="collapsed"
    )

    ask_button = st.button(
        "🚀 Ask AI"
    )


    # ========================================================
    # ANSWER
    # ========================================================

    if ask_button:

        if not question:

            st.warning("Please enter a question.")

        else:

            with st.spinner("🔎 Searching your document and generating answer..."):

                retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": 3}
                )

                # Retrieve relevant documents
                results = retriever.invoke(question)

                # Combine context
                context = "\n\n".join(
                    result.page_content
                    for result in results
                )

                # Prompt
                prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the
information provided in the context.

If the answer cannot be found in the context,
say:

"I couldn't find this information in the document."

Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""

                # Generate response
                response = llm.invoke(prompt)


            # ====================================================
            # DISPLAY ANSWER
            # ====================================================

            st.markdown("### 🤖 AI Answer")

            st.markdown(
                f'<div class="answer-box">{response.content}</div>',
                unsafe_allow_html=True
            )


            # ====================================================
            # SOURCES
            # ====================================================

            st.markdown("### 📄 Retrieved Sources")

            for i, result in enumerate(results):

                page = result.metadata.get(
                    "page",
                    "Unknown"
                )

                if isinstance(page, int):
                    page = page + 1

                with st.expander(
                    f"Source {i + 1}  •  Page {page}"
                ):

                    st.markdown(
                        f'<div class="source-box">{result.page_content}</div>',
                        unsafe_allow_html=True
                    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    '<div style="text-align:center; color:#737b8c;">'
    '🤖 Local RAG AI Assistant<br>'
    'Powered by LangChain • FAISS • Sentence Transformers • Ollama'
    '</div>',
    unsafe_allow_html=True
)
