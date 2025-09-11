"""
Askly - A chatbot for document information extraction and Q&A
"""

__version__ = "0.1.0"
__author__ = "Askly Team"

from .chatbot import AsklyBot
from .document_processor import DocumentProcessor
from .qa_engine import QAEngine

__all__ = ["AsklyBot", "DocumentProcessor", "QAEngine"]