import os
import shutil
import glob
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------------
# Page Configuration & ChatGPT UI Styling
# -----------------------------------
st.set_page_config(
    page_title="Uday Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* ChatGPT Dark & Sleek Modern Theme */
    .stApp {
        max-width: 1150px;
        margin: 0 auto;
    }
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 14px;
        padding: 1rem;
    }
    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
    }
    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #ffffff;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #a0aec0;
        font-size: 1.05rem;
    }
    .status-badge {
        background-color: #1a202c;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------
# Load Secret API Key from .env or Streamlit Cloud Secrets
# -----------------------------------
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key and hasattr(st, "secrets"):
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        pass

if not api_key:
    api_key = os.getenv("OPENAI_API_KEY", "")

if not api_key:
    st.error("❌ `GOOGLE_API_KEY` is missing. Please add it to Streamlit Secrets in the Cloud Settings.")
    st.stop()

# -----------------------------------
# Initialize Gemini Models
# -----------------------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=api_key
)

# -----------------------------------
# Data Loader Function (Loads files from data/ folder)
# -----------------------------------
def load_and_index_data():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    txt_files = glob.glob(os.path.join(data_dir, "*.txt")) + glob.glob(os.path.join(data_dir, "*.md"))
    all_files = pdf_files + txt_files
    
    if not all_files:
        return None, [], 0, 0

    documents = []
    loaded_filenames = []

    for file_path in pdf_files:
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
            loaded_filenames.append(os.path.basename(file_path))
        except Exception as e:
            st.sidebar.error(f"Error loading {os.path.basename(file_path)}: {e}")

    for file_path in txt_files:
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            documents.extend(docs)
            loaded_filenames.append(os.path.basename(file_path))
        except Exception as e:
            st.sidebar.error(f"Error loading {os.path.basename(file_path)}: {e}")

    if not documents:
        return None, [], 0, 0

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    # Build in-memory vectorstore for cloud compatibility
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectorstore, loaded_filenames, len(documents), len(chunks)


# -----------------------------------
# Session State & Indexing
# -----------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed" not in st.session_state:
    st.session_state.indexed = False

# Auto-index data on first startup if vectorstore not loaded
if not st.session_state.indexed:
    with st.spinner("⚡ Loading knowledge base from `data/` folder..."):
        vectorstore, filenames, total_docs, total_chunks = load_and_index_data()
        st.session_state.vectorstore = vectorstore
        st.session_state.filenames = filenames
        st.session_state.total_docs = total_docs
        st.session_state.total_chunks = total_chunks
        st.session_state.indexed = True

# -----------------------------------
# Sidebar - Uday Chatbot Controls
# -----------------------------------
with st.sidebar:
    st.title("🤖 Uday Chatbot")
    st.caption("AI Assistant trained on your custom data")
    st.divider()

    st.subheader("📚 Knowledge Base (`data/`)")
    
    if st.session_state.filenames:
        st.success(f"Loaded {len(st.session_state.filenames)} File(s)")
        for fn in st.session_state.filenames:
            st.markdown(f"📄 `{fn}`")
        
        st.markdown(f"**Total Pages/Docs:** `{st.session_state.total_docs}`")
        st.markdown(f"**Total Chunks:** `{st.session_state.total_chunks}`")
    else:
        st.warning("⚠️ No documents found in `data/` folder.")
        st.info("💡 Add your `.pdf` or `.txt` files to the `data/` folder and click **Sync Data** below.")

    st.divider()

    if st.button("🔄 Sync / Re-index Data", use_container_width=True, type="primary"):
        with st.spinner("Re-indexing `data/` folder..."):
            vectorstore, filenames, total_docs, total_chunks = load_and_index_data()
            st.session_state.vectorstore = vectorstore
            st.session_state.filenames = filenames
            st.session_state.total_docs = total_docs
            st.session_state.total_chunks = total_chunks
            st.session_state.indexed = True
            st.success("Re-indexing complete!")
            st.rerun()

    st.divider()

    st.subheader("⚙️ Settings")
    top_k = st.slider(
        "Context Chunks (k):",
        min_value=1,
        max_value=10,
        value=4,
        help="Number of data chunks retrieved per query"
    )

    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# -----------------------------------
# Main ChatGPT-style Interface
# -----------------------------------
st.markdown("""
    <div class="main-header">
        <h1>🤖 Uday Chatbot</h1>
        <p>Ask anything about the documents stored in your <code>data/</code> folder!</p>
    </div>
""", unsafe_allow_html=True)

# Display banner if data folder has no files
if not st.session_state.filenames:
    st.warning("📂 Place your PDFs or text files in the `data/` folder and click **Sync / Re-index Data** in the sidebar.")

# Render existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("🔍 View Referenced Sources"):
                for idx, doc in enumerate(message["sources"]):
                    page_info = f" (Page {doc.metadata.get('page', 0) + 1})" if "page" in doc.metadata else ""
                    st.markdown(f"**Source {idx + 1}{page_info}:**")
                    st.caption(doc.page_content)
                    st.divider()

# Chat Input Box
if user_prompt := st.chat_input("Ask Uday a question..."):
    
    # 1. Append User Message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # 2. Assistant Response
    with st.chat_message("assistant"):
        if not st.session_state.vectorstore:
            answer_text = "I don't have any data loaded yet. Please place files in the `data/` folder and click **Sync / Re-index Data**."
            st.markdown(answer_text)
            relevant_docs = []
        else:
            with st.spinner("Uday is thinking..."):
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": top_k})
                relevant_docs = retriever.invoke(user_prompt)

                context = "\n\n".join([doc.page_content for doc in relevant_docs])

                prompt = ChatPromptTemplate.from_template("""
You are Uday Chatbot, an intelligent, helpful, and polite AI assistant.

Answer the user's question accurately using ONLY the provided context below.

If the answer cannot be found in the context, politely state:
"I couldn't find this information in the uploaded data."

Context:
{context}

Question:
{question}

Answer:
""")
                formatted_prompt = prompt.format(context=context, question=user_prompt)
                response = llm.invoke(formatted_prompt)
                answer_text = response.content

                st.markdown(answer_text)

                if relevant_docs:
                    with st.expander("🔍 View Referenced Sources"):
                        for idx, doc in enumerate(relevant_docs):
                            page_info = f" (Page {doc.metadata.get('page', 0) + 1})" if "page" in doc.metadata else ""
                            st.markdown(f"**Source {idx + 1}{page_info}:**")
                            st.caption(doc.page_content)
                            st.divider()

    # Save to Session State
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "sources": relevant_docs
    })