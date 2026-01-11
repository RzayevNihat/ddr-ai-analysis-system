# DDR AI Analysis System

Automated analysis of **Daily Drilling Reports (DDR)** using artificial intelligence techniques such as NLP, large language models, and knowledge graphs.

## Overview

This system parses DDR PDF files, extracts structured operational data, detects drilling anomalies, builds a knowledge graph, and provides a RAG-based question answering interface through an interactive dashboard.

## Demo

[https://ddr-ai-analysis-system.streamlit.app/](https://ddr-ai-analysis-system.streamlit.app/)

## Features

* PDF parsing and metadata extraction
* Drilling activity classification
* Anomaly detection (stuck pipe, high gas, lost circulation)
* AI-generated daily summaries
* Knowledge graph for wells, depths, activities, and anomalies
* RAG-based question answering system
* Interactive Streamlit dashboard

## Architecture

```
ddr-ai-system/
├── data/
│   ├── pdfs/
│   └── processed/
├── chroma_db/
├── src/
│   ├── pdf_processor.py
│   ├── nlp_processor.py
│   ├── knowledge_graph.py
│   ├── llm_service.py
│   └── rag_system.py
├── app.py
├── setup_data.py
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone <repo-url>
cd ddr-ai-system
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
PDF_DATA_PATH=./data/pdfs
PROCESSED_DATA_PATH=./data/processed
CHROMA_PERSIST_DIR=./chroma_db
EMBEDDINGS_MODEL=all-MiniLM-L6-v2
```

## Usage

1. Add DDR PDF files to `data/pdfs/`
2. Process the data:

```bash
python setup_data.py
```

3. Run the application:

```bash
streamlit run app.py
```

## Example Queries

* Show all intervals with gas peaks above 1.2%
* List all stuck pipe events
* What activities were performed at 2800 m depth?

## Technology Stack

* PDF processing: pdfplumber
* NLP: spaCy
* Embeddings: SentenceTransformers
* Vector database: ChromaDB
* LLM: Groq (Llama 3.3 70B)
* Knowledge graph: NetworkX, Pyvis
* Frontend: Streamlit

## Troubleshooting

* PDF parsing issues: inspect PDF structure using pdfplumber
* Rate limit errors: verify Groq API key and limits
* ChromaDB errors: remove `chroma_db/` and rerun data setup

## Roadmap

* Predictive anomaly detection
* Multi-well comparative analysis
* Export reports to PDF and Excel
* Multi-language support

---

This project is designed for intelligent analysis of drilling operational data.
