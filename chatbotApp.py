import streamlit as st
import os
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions
from huggingface_hub import InferenceClient

# Configuration & Token
# Hugging Face API Token
HF_TOKEN = "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"

# Mistral-7B-v0.2
client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.2", token=HF_TOKEN)

st.set_page_config(page_title="RAG Data Viz Bot", page_icon="📊", layout="wide")

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text_chunks = []
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text_chunks.append(content)
    return text_chunks

# Word Embedding (SentenceTransformer) and ChromaDB
@st.cache_resource
def initialize_database():
    # Embedding Model: all-MiniLM-L6-v2 
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    # ChromaDB setup (In-memory database)
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="viz_guide_db", embedding_function=emb_fn)
    
    if os.path.exists("data_viz_guide.pdf"):
        chunks = extract_text_from_pdf("data_viz_guide.pdf")
        for i, chunk in enumerate(chunks):
            collection.add(documents=[chunk], ids=[f"id_{i}"])
        return collection
    else:
        st.error("Error: 'data_viz_guide.pdf' not found in GitHub repository!")
        return None

# System Load
with st.spinner("Initializing RAG System (Vector DB + Embedding)..."):
    db = initialize_database()

st.title("📊 Data Visualization Advisor")
st.markdown("Architecture: **RAG** | DB: **ChromaDB** | LLM: **Mistral-7B**")

# Chat History Setup
if "messages" not in st.session_state:
    st.session_state.messages = []

# Message History Shown
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("How can I help you with your charts?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if db:
            # K=10
            results = db.query(query_texts=[prompt], n_results=10)
            context_data = " ".join(results['documents'][0])
            
            # Prompt Engineering for LLM
            messages = [
                {"role": "system", "content": "You are a Data Viz Expert. Use the context to answer."},
                {"role": "user", "content": f"Context: {context_data}\n\nQuestion: {prompt}"}
            ]
            
            try:
                # Answer generataed from LLM
                response = client.chat_completion(messages=messages, max_tokens=512)
                answer = response.choices[0].message.content
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"LLM API Error: {str(e)}")
        else:
            st.error("Database not initialized.")

# Sidebar info
st.sidebar.info("This bot uses RAG (Retrieval-Augmented Generation) to answer questions from your PDF guide.")

