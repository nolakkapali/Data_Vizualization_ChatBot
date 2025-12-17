# 📊 Data Visualization Advisor Chatbot (RAG System)

This project is a specialized AI Chatbot designed to act as a **Data Visualization Advisor**. It uses a **Retrieval-Augmented Generation (RAG)** architecture to provide expert advice based on a data visualization chart suggesion guide. It basically recommends charts based on the user given scenarios. It includes vector databases such as chromaDB, word embeddings, and a large language model (LLM).

---

## Project Link

 **Streamlit Cloud:**     [Visit the Chatbot](https://datavizualizationchatbot-5xw4fm24pup9zk4ewbn2ym.streamlit.app/#data-visualization-advisor)


---

## 🚀 Key Features
* **Contextual Knowledge:** Answers questions based on the provided `data_viz_guide.pdf`.
* **Semantic Search:** Finds relevant information even if the exact words don't match.
* **No OCR Processing:** Directly extracts digital text from PDFs for high accuracy.
* **Parameter Control:** Allows users to adjust chunk size, Top-K retrieval.

---
## 🛠️ Development Tools

* **Python 3.12.8:** Stable environment for AI and data science libraries.
* **Streamlit:** For building and deploying the interactive web interface.
* **ChromaDB:** A high-performance vector database for storing document embeddings.
* **Hugging Face Hub:** To connect with Large Language Models (LLMs) via API.
* **Sentence-Transformers:** Used for generating high-quality word embeddings (all-MiniLM-L6-v2).
* **PyPDF:** For lightweight and direct text extraction from PDF files.

---

## 🛠️ Technical Specifications

* **LLM Model:**	Mistral-7B-Instruct-v0.2 (7 Billion Parameters, exceeds 1B+ req)
* **Vector Database:** ChromaDB (In-memory vector storage)
* **Embedding Model:**	all-MiniLM-L6-v2 (Professional Word Embedding)
* **PDF Extraction:**	PyPDF (Direct digital extraction - No OCR)
* **Top-K Retrieval:**	K=10 (Retrieves the 10 most relevant segments)
* **Hosting:**	Streamlit Cloud

---

## 🛠️ User Input
  Here are the type of questions you can ask to the chatbot.
* Which chart should I use to show a trend over time?
* If I want to compare five different categories, which chart is best?
* When should I use a Pie Chart instead of a Bar Chart?<br>
* What is the difference between a histogram and a bar chart according to the guide?
* Can you explain the best practices for choosing colors in data visualization?
* What are the common mistakes to avoid in a dashboard?
* Give me a detailed summary of the data visualization principles mentioned in the document.

---

## 🛠️ Project Workflow (Architecture)

* **Ingestion:** The PDF is loaded and split into small segments (Chunks) to maintain context.
* **Vectorization:** Each segment is converted into a vector using the all-MiniLM-L6-v2 embedding model.
* **Storage:** These vectors are stored in ChromaDB.
* **Retrieval:** When a user asks a question, the system searches the database for the top 10 (K=10) most similar text chunks.
* **Generation:** The retrieved chunks are provided to the LLM as "Context." The LLM generates a natural language answer based only on that verified context.

```mermaid
graph TD
    A[data_viz_guide.pdf] -->|PyPDF Parsing| B(Text Chunks)
    B -->|Sentence Transformers| C{Word Embedding Model}
    C -->|all-MiniLM-L6-v2| D[(ChromaDB Vector Store)]
    
    E[User Query] -->|Similarity Search| D
    D -->|Retrieve Top K=10| F[Relevant Context]
    
    F -->|Context + Query| G[Mistral-7B LLM]
    G -->|Final Response| H[Streamlit UI]
