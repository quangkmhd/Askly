"""
Dynamic prompts based on context
"""
from typing import List, Dict, Any, Optional


class DynamicPromptGenerator:
    """Generate prompts dynamically based on context"""
    
    def __init__(self):
        pass
    
    def generate_rag_prompt(
        self,
        query: str,
        context_items: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate dynamic RAG prompt based on context
        
        Args:
            query: User question
            context_items: Retrieved context chunks
            chat_history: Previous conversation
            
        Returns:
            Formatted prompt string
        """
        # Format context - TOP 3 ONLY with 600 char limit (OPTIMIZED)
        context_parts = []
        for i, item in enumerate(context_items[:3], 1):  # TOP 3 ONLY
            headers = item.get('headers', [])
            header_text = ' > '.join(headers) if headers else ''
            page = item.get('page_number', 'N/A')
            
            # Limit context length to 600 chars to reduce noise
            text = item['sentence_chunk'][:600]
            
            context_str = f"[{i}]"
            if header_text:
                context_str += f" {header_text}"
            context_str += f" (Trang {page})\n{text}"
            
            context_parts.append(context_str)
        
        context = "\n\n".join(context_parts)
        
        # Add chat history context if available
        history_context = ""
        if chat_history and len(chat_history) > 0:
            # Get last user question for context
            last_messages = chat_history[-2:] if len(chat_history) >= 2 else chat_history
            for msg in last_messages:
                if msg['role'] == 'user':
                    history_context = f"\n(Câu hỏi trước: {msg['content']})"
                    break
        
        # IMPROVED PROMPT: Natural, conversational responses
        prompt = f"""Bạn là trợ lý tư vấn của trường Đại học FPT. Trả lời câu hỏi dựa vào tài liệu dưới đây.

YÊU CẦU:
- Trả lời bằng câu văn tự nhiên, liền mạch (KHÔNG dùng bullet points)
- Ngắn gọn, dễ hiểu, thân thiện
- Trích dẫn chính xác thông tin từ tài liệu{history_context}

TÀI LIỆU:
{context}

CÂU HỎI: {query}
TRẢ LỜI:"""
        
        return prompt


if __name__ == "__main__":
    # Test
    generator = DynamicPromptGenerator()
    
    context_items = [
        {
            'sentence_chunk': 'Học phí năm 1: 31.600.000đ/kỳ',
            'headers': ['C. Học phí', '1. Nhóm ngành CNTT'],
            'source_file': 'Học vụ.pdf',
            'page_number': 3
        }
    ]
    
    query = "Học phí năm 1 là bao nhiêu?"
    
    prompt = generator.generate_rag_prompt(query, context_items)
    
    print(prompt)
