# AI Chat Assistant for Website and PDF Queries

An intelligent chatbot interface built with **LangChain**, **Ollama**, and **Streamlit**, enabling users to ask context-aware questions on the fly from **uploaded PDF documents** or **website content**. It uses **Retrieval-Augmented Generation (RAG)** and **FAISS** vector search for fast and accurate answers grounded in real data.

---

## 🔍 Features

- 📄 Upload and query **PDF files**
- 🌐 Enter a **web URL** and ask questions from the page content
- 🔍 Implements **RAG (Retrieval-Augmented Generation)** for accurate answers
- 🧠 Switch between **LLMs** like `llama3` and `gemma` using **Ollama**(comparision between two)
- ⚡ Uses **FAISS** and **HuggingFace embeddings** for semantic search
- 💬 Built with **Streamlit** for an interactive chatbot experience
- ⏱️ Tracks **response time** for each query

---
##  Example Results

### 📄 PDF Upload Example
![PDF query 1](https://github.com/user-attachments/assets/fa19614c-da8e-493d-8cfb-0ca92b4fd76f)


---

### 🌐 Web URL Example
![Web Base ](https://github.com/user-attachments/assets/24a5824f-6df1-4c10-b893-846daff14d07)




---

## 📦 Tech Stack

| Tool/Library       | Purpose                                |
|--------------------|----------------------------------------|
| LangChain          | RAG pipeline & document processing     |
| Ollama             | Local LLMs (e.g., LLaMA 3, Gemma)       |
| FAISS              | Vector similarity search               |
| HuggingFace        | Text embeddings via MiniLM             |
| Streamlit          | Web-based UI for user interaction      |
| PyPDFLoader        | Parsing uploaded PDFs                  |
| WebBaseLoader      | Scraping content from live websites    |

---

