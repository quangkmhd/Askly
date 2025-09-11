"""
Main chatbot class that integrates document processing and Q&A capabilities.
"""

import os
from typing import Dict, List, Any, Optional
from .document_processor import DocumentProcessor
from .qa_engine import QAEngine


class AsklyBot:
    """
    Main chatbot class for document-based Q&A and information extraction.
    """
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.qa_engine = QAEngine()
        self.loaded_documents = []
        self.is_ready = False
    
    def load_document(self, file_path: str) -> Dict[str, Any]:
        """
        Load a document into the chatbot's knowledge base.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document information and loading status
        """
        try:
            # Process the document
            document = self.document_processor.load_document(file_path)
            
            # Preprocess the content
            document['content'] = self.document_processor.preprocess_text(document['content'])
            
            # Add to Q&A engine
            self.qa_engine.add_document(document)
            
            # Store document info
            self.loaded_documents.append({
                'filename': document['filename'],
                'file_type': document['file_type'],
                'word_count': document['word_count'],
                'char_count': document['char_count']
            })
            
            # Rebuild index
            self.qa_engine.build_index()
            self.is_ready = True
            
            return {
                'status': 'success',
                'message': f"Đã tải thành công tài liệu: {document['filename']}",
                'document_info': document
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Lỗi khi tải tài liệu: {str(e)}",
                'document_info': None
            }
    
    def load_documents_from_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory_path: Path to the directory containing documents
            
        Returns:
            Summary of loaded documents
        """
        if not os.path.exists(directory_path):
            return {
                'status': 'error',
                'message': f"Thư mục không tồn tại: {directory_path}",
                'loaded_files': []
            }
        
        loaded_files = []
        failed_files = []
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            if os.path.isfile(file_path):
                result = self.load_document(file_path)
                
                if result['status'] == 'success':
                    loaded_files.append(filename)
                else:
                    failed_files.append({
                        'filename': filename,
                        'error': result['message']
                    })
        
        return {
            'status': 'success' if loaded_files else 'error',
            'message': f"Đã tải {len(loaded_files)} tài liệu thành công",
            'loaded_files': loaded_files,
            'failed_files': failed_files
        }
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get an answer based on loaded documents.
        
        Args:
            question: User's question
            
        Returns:
            Answer and related information
        """
        if not self.is_ready:
            return {
                'status': 'error',
                'answer': 'Vui lòng tải ít nhất một tài liệu trước khi đặt câu hỏi.',
                'confidence': 0.0,
                'sources': []
            }
        
        try:
            result = self.qa_engine.answer_question(question)
            
            return {
                'status': 'success',
                'question': question,
                'answer': result['answer'],
                'confidence': result['confidence'],
                'sources': result['sources'],
                'relevant_chunks': result['relevant_chunks']
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'answer': f'Lỗi khi xử lý câu hỏi: {str(e)}',
                'confidence': 0.0,
                'sources': []
            }
    
    def extract_information(self, query: str) -> Dict[str, Any]:
        """
        Extract specific information from loaded documents.
        
        Args:
            query: Information extraction query
            
        Returns:
            Extracted information
        """
        if not self.is_ready:
            return {
                'status': 'error',
                'message': 'Vui lòng tải ít nhất một tài liệu trước khi trích xuất thông tin.',
                'results': []
            }
        
        try:
            results = self.qa_engine.extract_information(query)
            
            return {
                'status': 'success',
                'query': query,
                'results': results,
                'total_results': len(results)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Lỗi khi trích xuất thông tin: {str(e)}',
                'results': []
            }
    
    def get_document_summary(self, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary of loaded documents or a specific document.
        
        Args:
            filename: Optional specific filename to get summary for
            
        Returns:
            Document summary information
        """
        if not self.loaded_documents:
            return {
                'status': 'error',
                'message': 'Chưa có tài liệu nào được tải.',
                'summary': {}
            }
        
        if filename:
            # Get summary for specific document
            doc = next((d for d in self.loaded_documents if d['filename'] == filename), None)
            if not doc:
                return {
                    'status': 'error',
                    'message': f'Không tìm thấy tài liệu: {filename}',
                    'summary': {}
                }
            return {
                'status': 'success',
                'summary': doc
            }
        else:
            # Get summary for all documents
            stats = self.qa_engine.get_statistics()
            
            return {
                'status': 'success',
                'summary': {
                    'total_documents': len(self.loaded_documents),
                    'total_chunks': stats['total_chunks'],
                    'total_words': stats['total_words'],
                    'loaded_documents': self.loaded_documents
                }
            }
    
    def get_document_keywords(self, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract keywords from loaded documents.
        
        Args:
            filename: Optional specific filename to get keywords for
            
        Returns:
            Keywords from documents
        """
        if not self.loaded_documents:
            return {
                'status': 'error',
                'message': 'Chưa có tài liệu nào được tải.',
                'keywords': []
            }
        
        # For now, return a simple implementation
        # In a full implementation, you might want to store and process keywords separately
        return {
            'status': 'success',
            'message': 'Tính năng trích xuất từ khóa sẽ được cải thiện trong phiên bản tương lai.',
            'keywords': []
        }
    
    def clear_documents(self) -> Dict[str, Any]:
        """
        Clear all loaded documents from memory.
        
        Returns:
            Operation status
        """
        self.loaded_documents.clear()
        self.qa_engine = QAEngine()  # Reset the Q&A engine
        self.is_ready = False
        
        return {
            'status': 'success',
            'message': 'Đã xóa tất cả tài liệu khỏi bộ nhớ.'
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current chatbot status.
        
        Returns:
            Current status information
        """
        return {
            'is_ready': self.is_ready,
            'total_documents': len(self.loaded_documents),
            'supported_formats': self.document_processor.supported_formats,
            'loaded_documents': [doc['filename'] for doc in self.loaded_documents]
        }