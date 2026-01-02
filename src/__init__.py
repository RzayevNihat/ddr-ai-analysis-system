# src/__init__.py
"""
DDR AI Analysis System
Core modules for PDF processing, NLP, and RAG
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Make imports easier
from .config import Config
from .llm_service import LLMService
from .rag_system import RAGSystem
from .knowledge_graph import KnowledgeGraph
from .nlp_processor import NLPProcessor
from .pdf_processor import DDRParser

__all__ = [
    'Config',
    'LLMService',
    'RAGSystem',
    'KnowledgeGraph',
    'NLPProcessor',
    'DDRParser'
]