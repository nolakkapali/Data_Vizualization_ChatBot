import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain_community.chains import RetrievalQA

#Configuration & Token
#Hugging Face token is used here so that it can be run on the cloud
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

st.set_page_config(page_title="Data Viz RAG Bot", page_icon="📊", layout="wide")

#Sidebar Parameter for parameters controlling
st.sidebar.header("RAG Parameters")
st.sidebar.info("Adjust these settings to control the RAG pipeline.")

chunk_size = st.sidebar.slider("Chunk Size", min_value=100, max_value=1000, value=500, help="Size of each document segment.")
chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=0, max_value=200, value=50)
k_value = st.sidebar.slider("Top-K (Retrieval)", min_value=1, max_value=15, value=10, help="Value is K=10")
temperature = st.sidebar.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=0.1)

st.title("Data Visualization Advisor Chatbot")
st.markdown(f"This system uses **Gemma-2B (1B+ LLM)**, **Chroma DB**, and **RAG** architecture.")

#Docs to Vectors & Chroma DB
@st.cache_resource
def initialize_rag_system(_c_size, _c_overlap, _k, _temp):
    #here it directly reads text from pdf
    loader = PyPDFLoader("data_viz_guide.pdf")
    documents = loader.load()
    
    #Text is splitted here
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=_c_size, chunk_overlap=_c_overlap)
    chunks = text_splitter.split_documents(documents)
    
    #Embedding Model is all-MiniLM-L6-v2
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    #Chroma DB is use here for vector database
    vector_db = Chroma.from_documents(chunks, embeddings)
    
    # LLM Model is Gemma-2B( here uses 2 billion parameter)
    llm = HuggingFaceHub(
        repo_id="google/gemma-2b", 
        model_kwargs={"temperature": _temp, "max_length": 500}
    )
    
    # RetrievalQA Chain
    return RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=vector_db.as_retriever(search_kwargs={"k": _k})
    )

#System load
with st.spinner("Processing documents and setting up Vector DB..."):
    qa_chain = initialize_rag_system(chunk_size, chunk_overlap, k_value, temperature)

#chat_UI
if "messages" not in st.session_state:
    st.session_state.messages = []

#Showing history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#Questions are taken from user
if prompt := st.chat_input("Ask a question (e.g., Which chart for trends?)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        #Replies or answers are made from RAG process
        response = qa_chain.run(prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

#Footer
st.sidebar.markdown("---")
st.sidebar.write("**Docs to Vectors:** PyPDFLoader")
st.sidebar.write("**Vector DB:** Chroma DB")

st.sidebar.write("**LLM:** Gemma-2B (Cloud)")
