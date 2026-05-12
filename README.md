# ATS Resume Builder

ATS Resume Builder is a RAG-based application that generates ATS-friendly resumes based on a candidate’s uploaded resume and a target job description.

The application uses Retrieval-Augmented Generation (RAG) with LangChain, ChromaDB, and Google Gemini API to retrieve relevant resume information and optimize the resume according to the job role and ATS keywords.

---

## Live Demo

🚀 Streamlit Application:  
https://resumebuilder-chaxqsufepbdfyb6tappf2z.streamlit.app/

---

## Features

- Upload Resume PDF
- Enter Job Description
- ATS-Friendly Resume Generation
- Role-Based Resume Optimization
- RAG Pipeline Implementation
- Semantic Retrieval using Embeddings
- ChromaDB Vector Database
- Google Gemini API Integration
- Streamlit Web Application

---

## Tech Stack

- Python
- Streamlit
- LangChain
- ChromaDB
- Google Gemini API
- RAG (Retrieval-Augmented Generation)
- Vector Embeddings

---

## Project Workflow

1. Upload Resume PDF
2. Extract Resume Text
3. Split Resume into Chunks
4. Generate Embeddings
5. Store Embeddings in ChromaDB
6. Retrieve Relevant Resume Sections
7. Generate ATS-Optimized Resume using Gemini LLM

---

## RAG Architecture

Resume PDF  
↓  
Document Loader  
↓  
Text Splitter  
↓  
Embedding Model  
↓  
ChromaDB Vector Store  
↓  
Retriever  
↓  
Prompt Engineering  
↓  
Gemini LLM  
↓  
ATS-Friendly Resume Output

---
