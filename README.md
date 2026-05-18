# 📚 E-Library Intelligent Search System

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Framework](https://img.shields.io/badge/framework-Streamlit-FF4B4B)
![LLM](https://img.shields.io/badge/LLM-Llama--3-orange)

An advanced, multimodal Information Retrieval (IR) system for an E-Library. This project features hybrid text search, visual cover-based search, personalized recommendations, and a conversational AI librarian powered by Retrieval-Augmented Generation (RAG).

## ✨ Features

- **Hybrid Search Engine**: Combines Vector Space Model (TF-IDF) and Semantic Search (Sentence Transformers) for highly relevant text-based results.
- **Visual Search**: Multimodal capabilities allowing users to search books based on visual characteristics of their covers (e.g., "blue book cover", "minimalist design").
- **Multimodal RAG Assistant**: An intelligent Chat Assistant (using Groq API and Llama-3) that understands both text and visual queries, providing personalized recommendations in natural language.
- **Personalization Engine**: Adapts search results dynamically based on user interactions, session history, and bookmarked items.
- **Query Expansion**: Automatically broadens search queries with synonyms and related terms to improve recall.
- **Interactive Dashboard**: A beautiful, fully-responsive dark-themed Streamlit web interface with real-time analytics.
- **Comprehensive Evaluation**: Extensive evaluation scripts generating metrics like Precision@K, Recall@K, MRR, and NDCG.

## 📂 Project Structure

```text
├── src/                                  # Core application logic
│   ├── app_advanced.py                   # Main Streamlit web application
│   ├── search_engine_advanced.py         # Hybrid search & personalization logic
│   ├── rag_llm_multimodal.py             # RAG conversational assistant
│   ├── multimodal_search_kaggle.py       # Visual & multimodal search engine
│   └── evaluate_comprehensive.py         # Evaluation and metrics generation
├── data/                                 # Datasets and pre-computed models
│   ├── books.csv                         # Text search dataset
│   └── main_dataset.csv                  # Multimodal dataset
├── results/                              # Evaluation metrics and performance plots
├── report/                               # Comprehensive project report (PDF)
└── .gitignore                            # Git ignore file (excludes book covers)
```

## 🚀 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Varunbhawnani/ir_project_12345.git
cd ir_project_12345
```

### 2. Install Dependencies
Make sure you have Python 3.9+ installed. You can install the required packages using pip:
```bash
pip install streamlit pandas numpy plotly sentence-transformers groq scikit-learn
```

### 3. Environment Variables
To use the conversational AI features, you need a Groq API Key. Export it in your environment:
```bash
export GROQ_API_KEY="your-groq-api-key"
```

### 4. Run the Application
Start the Streamlit application:
```bash
streamlit run src/app_advanced.py
```

## 🧠 Architecture Overview

- **Frontend Application**: Built using **Streamlit** with custom CSS for a premium, interactive UI.
- **Text Embeddings**: **Sentence Transformers** (`all-MiniLM-L6-v2`) used for semantic understanding.
- **Visual Embeddings**: CLIP-based or custom feature extraction for book cover matching.
- **LLM Integration**: **Llama-3.3-70b-versatile** via the **Groq API** for ultra-fast RAG inference.
- **Indexing & Storage**: Custom-built in-memory dense matrix operations (via NumPy/SciPy) for fast retrieval.

## 📊 Evaluation & Metrics
The system has been thoroughly evaluated against standard IR metrics. You can find detailed charts comparing VSM vs. Semantic vs. Hybrid approaches in the `results/` directory, including:
- Overall System Comparison
- Text Search Metrics
- Visual Search Performance
- RAG System Evaluation

To run the evaluation suite yourself:
```bash
python src/evaluate_comprehensive.py
```

---
*Developed as part of an Information Retrieval Academic Project.*
