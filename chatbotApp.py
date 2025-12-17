import os
import streamlit as st
from pypdf import PdfReader
from huggingface_hub import InferenceClient

# Configuration
HF_TOKEN = "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"
client = InferenceClient(model="google/gemma-2b-it", token=HF_TOKEN)

st.set_page_config(page_title="Easy Data Bot", page_icon="📊")

# ১. PDF থেকে টেক্সট পড়ার সহজ ফাংশন
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

st.title("Data Visualization Advisor")

# সিস্টেম লোড
if os.path.exists("data_viz_guide.pdf"):
    with st.spinner("Reading PDF Guide..."):
        context_text = extract_text_from_pdf("data_viz_guide.pdf")
else:
    st.error("PDF file not found! Please upload 'data_viz_guide.pdf' to GitHub.")
    st.stop()

# Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me about charts"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # সরাসরি Hugging Face-কে প্রশ্ন পাঠানো (Prompt Engineering)
        full_prompt = f"Context: {context_text[:3000]}\n\nQuestion: {prompt}\n\nAnswer the question based on the context above."
        
        response = client.text_generation(full_prompt, max_new_tokens=500)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
