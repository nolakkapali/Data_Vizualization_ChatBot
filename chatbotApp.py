import streamlit as st
import os
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions
from huggingface_hub import InferenceClient

#LLM Model (Gemma-2B) and Token
HF_TOKEN = "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"
client = InferenceClient(model="google/gemma-2b-it", token=HF_TOKEN)

st.set_page_config(page_title="RAG Chatbot", page_icon="📊")

#Pypdf for pdf without OCR
def load_pdf(file_path):
    reader = PdfReader(file_path)
    text_chunks = []
    for page in reader.pages:
        content = page.extract_text()
        if content:
            #Divided into smaller chunks
            text_chunks.append(content)
    return text_chunks

#Word Embedding & ChromaDB
@st.cache_resource
def setup_db():
    #Embedding Model: all-MiniLM-L6-v2
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    # ChromaDB setup
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="my_pdf_db", embedding_function=emb_fn)
    
    chunks = load_pdf("data_viz_guide.pdf")
    #Data stored in Database
    for i, chunk in enumerate(chunks):
        collection.add(documents=[chunk], ids=[f"id_{i}"])
    return collection

db = setup_db()

st.title("Data Viz RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        #Here K=10
        results = db.query(query_texts=[prompt], n_results=10)
        context = " ".join(results['documents'][0])
        
        #Input given to LLM
        final_prompt = f"Context: {context}\n\nQuestion: {prompt}\nAnswer:"
        response = client.text_generation(final_prompt, max_new_tokens=500)
        
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
