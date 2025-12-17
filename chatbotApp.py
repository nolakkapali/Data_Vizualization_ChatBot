import os
import streamlit as st

# এরর এড়াতে ওএস এনভায়রনমেন্ট সেটআপ
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# সঠিক ইমপোর্ট পাথসমূহ (Python 3.12 এর জন্য)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Page Config
st.set_page_config(page_title="Data Viz RAG Bot", page_icon="📊", layout="wide")

# Sidebar
st.sidebar.header("RAG Parameters")
chunk_size = st.sidebar.slider("Chunk Size", 100, 1000, 500)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 200, 50)
k_value = st.sidebar.slider("Top-K (Retrieval)", 1, 15, 10)
temperature = st.sidebar.slider("LLM Temperature", 0.0, 1.0, 0.1)

st.title("Data Visualization Advisor Chatbot")
st.markdown("This system uses **Gemma-2B**, **Chroma DB**, and **RAG** architecture.")

# RAG System Initialization
@st.cache_resource
def initialize_rag_system(_c_size, _c_overlap, _k, _temp):
    # PDF Load
    loader = PyPDFLoader("data_viz_guide.pdf")
    documents = loader.load()
    
    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=_c_size, chunk_overlap=_c_overlap)
    chunks = text_splitter.split_documents(documents)
    
    # Embeddings (Stable path)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(chunks, embeddings)
    
    # LLM (Stable path)
    llm = HuggingFaceHub(
        repo_id="google/gemma-2b", 
        model_kwargs={"temperature": _temp, "max_new_tokens": 512}
    )
    
    # Modern Prompt Template
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on the context.
    Context: {context}
    Question: {input}
    Answer:""")

    # Modern Chain Construction
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(vector_db.as_retriever(search_kwargs={"k": _k}), combine_docs_chain)
    
    return retrieval_chain

# Load System
with st.spinner("Setting up system..."):
    qa_chain = initialize_rag_system(chunk_size, chunk_overlap, k_value, temperature)

# Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask about charts..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # invoke() পদ্ধতি ব্যবহার করা হয়েছে
        response = qa_chain.invoke({"input": user_input})
        answer = response["answer"]
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
