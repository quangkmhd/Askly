"""
Query processing utilities: standalone question generation
"""
from typing import List, Dict, Optional
import re


class QueryProcessor:
    """Process user queries: generate standalone questions"""
    
    def __init__(self):
        pass
    
    def generate_standalone_question(
        self,
        current_query: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate standalone question from current query + chat history
        
        Args:
            current_query: Current user question
            chat_history: List of previous messages [{'role': 'user'/'assistant', 'content': '...'}]
            
        Returns:
            Standalone question with full context
        """
        if not chat_history or len(chat_history) == 0:
            return current_query
        
        # Check if current query needs context
        needs_context = self._needs_context(current_query)
        
        if not needs_context:
            return current_query
        
        # Extract context from history
        context = self._extract_context_from_history(chat_history)
        
        # Generate standalone question
        standalone = self._merge_context_with_query(context, current_query)
        
        return standalone
    
    def _needs_context(self, query: str) -> bool:
        """Check if query needs context from history"""
        query_lower = query.lower()
        
        # Indicators that query needs context
        context_indicators = [
            # Pronouns
            'nó', 'đó', 'này', 'kia', 'ấy',
            # Continuation words
            'còn', 'thì', 'thế', 'vậy', 'sao',
            # Comparative
            'hơn', 'khác', 'so với',
            # Short queries (likely follow-up)
        ]
        
        # Check for indicators
        has_indicator = any(ind in query_lower for ind in context_indicators)
        
        # Check if query is very short (< 3 words) - lowered from 5 to reduce false positives
        is_short = len(query.split()) < 3
        
        return has_indicator or is_short
    
    def _extract_context_from_history(
        self,
        chat_history: List[Dict[str, str]]
    ) -> Dict[str, str]:
        """Extract relevant context from chat history"""
        context = {
            'topic': None,
            'entity': None,
            'last_question': None
        }
        
        # Get last few messages (most recent first)
        recent_messages = chat_history[-6:]  # Last 3 turns (6 messages)
        
        # Extract topic from last user question
        for msg in reversed(recent_messages):
            if msg.get('role') == 'user':
                context['last_question'] = msg.get('content', '')
                # Try to extract topic
                topic = self._extract_topic(msg.get('content', ''))
                if topic:
                    context['topic'] = topic
                break
        
        return context
    
    def _extract_topic(self, text: str) -> Optional[str]:
        """Extract main topic from text"""
        text_lower = text.lower()
        
        # Topic keywords - simplified without intent classification
        topics = {
            'học phí': ['học phí', 'chi phí', 'tiền học'],
            'tuyển sinh': ['tuyển sinh', 'nhập học', 'đăng ký'],
            'điểm': ['điểm', 'gpa', 'điểm số'],
            'ngành học': ['ngành', 'chuyên ngành', 'chương trình'],
            'thời gian': ['thời gian', 'lịch', 'học kỳ'],
            'nghiên cứu': ['nghiên cứu', 'nckh', 'đề tài', 'khoa học'],
            'thi cử': ['thi', 'kiểm tra', 'nội quy thi'],
            'ký túc xá': ['ký túc xá', 'ktx', 'chỗ ở'],
            'thực tập': ['ojt', 'thực tập', 'đồ án'],
            'khen thưởng': ['khen thưởng', 'học bổng', 'giải thưởng'],
            'thủ tục': ['thủ tục', 'hồ sơ', 'giấy tờ'],
            'thạc sĩ': ['thạc sĩ', 'sau đại học', 'cao học'],
        }
        
        for topic, keywords in topics.items():
            if any(kw in text_lower for kw in keywords):
                return topic
        
        return None
    
    def _merge_context_with_query(
        self,
        context: Dict[str, str],
        current_query: str
    ) -> str:
        """Merge context with current query to create standalone question"""
        query_lower = current_query.lower()
        
        # If query starts with continuation words, merge with topic
        continuation_starters = ['còn', 'thì', 'thế', 'vậy', 'sao']
        
        if any(query_lower.startswith(word) for word in continuation_starters):
            if context['topic']:
                # Replace continuation word with topic
                # E.g., "còn năm 2 thì sao?" → "Học phí năm 2 thì sao?"
                standalone = f"{context['topic']} {current_query}"
                return standalone
        
        # If query has pronouns, try to resolve
        if any(pronoun in query_lower for pronoun in ['nó', 'đó', 'này', 'ấy']):
            if context['topic']:
                # Replace pronoun with topic
                standalone = current_query
                for pronoun in ['nó', 'đó', 'này', 'ấy']:
                    standalone = re.sub(
                        rf'\b{pronoun}\b',
                        context['topic'],
                        standalone,
                        flags=re.IGNORECASE
                    )
                return standalone
        
        # If query is very short, prepend topic
        if len(current_query.split()) < 5 and context['topic']:
            standalone = f"{context['topic']}: {current_query}"
            return standalone
        
        # Default: return original query
        return current_query


if __name__ == "__main__":
    # Test
    processor = QueryProcessor()
    
    # Test standalone question generation
    print("=== Standalone Question Generation ===")
    
    history = [
        {'role': 'user', 'content': 'Học phí của trường là bao nhiêu?'},
        {'role': 'assistant', 'content': 'Học phí năm 1 là 31.6M/kỳ'},
        {'role': 'user', 'content': 'còn năm 2 thì sao?'},
    ]
    
    current = "còn năm 2 thì sao?"
    standalone = processor.generate_standalone_question(current, history)
    
    print(f"Current: {current}")
    print(f"Standalone: {standalone}")
