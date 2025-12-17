import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceHub
# নতুন এবং সঠিক ইমপোর্ট পদ্ধতি
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Configuration & Token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

st.set_page_config(page_title="Data Viz RAG Bot", page_icon="📊", layout="wide")

# Sidebar Parameter
st.sidebar.header("RAG Parameters")
st.sidebar.info("Adjust these settings to control the RAG pipeline.")

chunk_size = st.sidebar.slider("Chunk Size", min_value=100, max_value=1000, value=500)
chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=0, max_value=200, value=50)
k_value = st.sidebar.slider("Top-K (Retrieval)", min_value=1, max_value=15, value=10)
temperature = st.sidebar.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=0.1)

st.title("Data Visualization Advisor Chatbot")
st.markdown(f"This system uses **Gemma-2B (LLM)**, **Chroma DB**, and **RAG** architecture.")

# Docs to Vectors & Chroma DB
@st.cache_resource
def initialize_rag_system(_c_size, _c_overlap, _k, _temp):
    loader = PyPDFLoader("data_viz_guide.pdf")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=_c_size, chunk_overlap=_c_overlap)
    chunks = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(chunks, embeddings)
    
    llm = HuggingFaceHub(
        repo_id="google/gemma-2b", 
        model_kwargs={"temperature": _temp, "max_new_tokens": 512}
    )
    
    # Modern RAG Chain Setup
    prompt = ChatPromptTemplate.from_template("""
    You are a Data Visualization Assistant. Use the following context to answer the question. 
    If you don't know the answer, just say you don't know.
    
    Context: {context}
    Question: {input}
    Answer:""")

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(vector_db.as_retriever(search_kwargs={"k": _k}), combine_docs_chain)
    
    return retrieval_chain

# System load
with st.spinner("Processing documents and setting up Vector DB..."):
    qa_chain = initialize_rag_system(chunk_size, chunk_overlap, k_value, temperature)

# Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question (e.g., Which chart for trends?)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # নতুন চেইন অনুযায়ী রেসপন্স পাওয়ার নিয়ম
        response = qa_chain.invoke({"input": prompt})
        answer = response["answer"]
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

# Footer
st.sidebar.markdown("---")
st.sidebar.write("**Vector DB:** Chroma DB")
st.sidebar.write("**LLM:** Gemma-2B (Cloud)")
