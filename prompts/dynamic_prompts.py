"""
Dynamic prompts based on intent and context
Few-shot examples for better accuracy
"""
from typing import List, Dict, Any, Optional


class DynamicPromptGenerator:
    """Generate prompts dynamically based on intent"""
    
    def __init__(self):
        # Few-shot examples for each intent
        self.few_shot_examples = {
            'tuition_fee': [
                {
                    'context': 'Học kỳ 1-3 (năm 1): 31.600.000đ/kỳ. Học kỳ 4-6 (năm 2): 33.600.000đ/kỳ.',
                    'question': 'Học phí là bao nhiêu?',
                    'answer': 'Học phí năm 1 là 31.600.000đ/kỳ, năm 2 là 33.600.000đ/kỳ.'
                },
                {
                    'context': 'Học kỳ 1-3: 31.600.000đ/kỳ. Tổng toàn khóa: 290-300 triệu đồng.',
                    'question': 'Học phí năm 1 là bao nhiêu?',
                    'answer': 'Học phí năm 1 là 31.600.000đ/kỳ.'
                },
                {
                    'context': 'Tổng học phí toàn khóa (9 kỳ): khoảng 290-300 triệu đồng.',
                    'question': 'Tổng học phí toàn khóa là bao nhiêu?',
                    'answer': 'Tổng học phí toàn khóa là khoảng 290-300 triệu đồng (9 kỳ).'
                }
            ],
            'admission': [
                {
                    'context': 'Yêu cầu: Top50 School Rank (21 điểm tổ hợp)',
                    'question': 'Điều kiện tuyển sinh là gì?',
                    'answer': 'Yêu cầu Top50 School Rank, tương đương khoảng 21 điểm tổ hợp.'
                }
            ],
            'grades': [
                {
                    'context': 'GPA >= 3.0 để được xếp loại Khá',
                    'question': 'GPA bao nhiêu để xếp loại Khá?',
                    'answer': 'GPA >= 3.0 để được xếp loại Khá.'
                }
            ]
        }
        
        # Intent-specific instructions
        self.intent_instructions = {
            'tuition_fee': """
QUY TẮC ĐẶC BIỆT CHO HỌC PHÍ (BẮT BUỘC):
1. TRÍCH DẪN CHÍNH XÁC số tiền từ tài liệu (VD: 31.600.000đ)
2. KHÔNG làm tròn, KHÔNG chuyển đổi đơn vị (giữ nguyên triệu/tỷ như trong tài liệu)
3. Nêu rõ: đồng/kỳ, đồng/năm, hoặc tổng toàn khóa
4. Phân biệt rõ từng năm học và nhóm ngành
5. TUYỆT ĐỐI KHÔNG bịa số liệu hoặc tính toán sai
6. Nếu hỏi "học phí là bao nhiêu", trả lời theo KỲ HỌC (31.600.000đ/kỳ), KHÔNG trả lời tổng toàn khóa trừ khi được hỏi cụ thể
""",
            'admission': """
QUY TẮC ĐẶC BIỆT CHO TUYỂN SINH:
1. Liệt kê đầy đủ các phương thức xét tuyển
2. Nêu rõ điều kiện cụ thể (điểm số, chứng chỉ)
3. Đề cập thời gian nếu có
""",
            'grades': """
QUY TẮC ĐẶC BIỆT CHO ĐIỂM SỐ:
1. Nêu rõ thang điểm (4.0, 10.0)
2. Phân biệt GPA, điểm môn học, điểm xếp loại
3. Trích dẫn chính xác ngưỡng điểm
""",
            'general': """
QUY TẮC CHUNG:
1. Trả lời dựa hoàn toàn vào tài liệu
2. Ngắn gọn, chính xác
3. Không bịa đặt thông tin
"""
        }
    
    def generate_rag_prompt(
        self,
        query: str,
        context_items: List[Dict[str, Any]],
        intent: str = 'general',
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate dynamic RAG prompt based on intent
        
        Args:
            query: User question
            context_items: Retrieved context chunks
            intent: Classified intent
            chat_history: Previous conversation
            
        Returns:
            Formatted prompt string
        """
        # Format context
        context_parts = []
        for i, item in enumerate(context_items, 1):
            headers = item.get('headers', [])
            header_text = ' > '.join(headers) if headers else ''
            source = item.get('source_file', 'N/A')
            page = item.get('page_number', 'N/A')
            
            context_str = f"[Context {i}]"
            if header_text:
                context_str += f" {header_text}"
            context_str += f" (Nguồn: {source}, Trang {page})\n"
            context_str += item['sentence_chunk']
            
            context_parts.append(context_str)
        
        context = "\n\n".join(context_parts)
        
        # Get intent-specific instructions
        instructions = self.intent_instructions.get(intent, self.intent_instructions['general'])
        
        # Get few-shot examples
        examples = self._format_few_shot_examples(intent)
        
        # Format chat history
        history_text = ""
        if chat_history and len(chat_history) > 0:
            history_parts = []
            for msg in chat_history[-6:]:
                if msg.get('role') == 'user':
                    history_parts.append(f"Người dùng: {msg.get('content', '')}")
                elif msg.get('role') == 'assistant':
                    history_parts.append(f"Trợ lý: {msg.get('content', '')}")
            if history_parts:
                history_text = f"\n\nLỊCH SỬ HỘI THOẠI:\n" + "\n".join(history_parts) + "\n"
        
        # Build prompt theo format Vi-Qwen2-RAG
        prompt = f"""Chú ý các yêu cầu sau:
- Câu trả lời phải chính xác và đầy đủ nếu ngữ cảnh có câu trả lời. 
- Chỉ sử dụng các thông tin có trong ngữ cảnh được cung cấp.
- Chỉ cần từ chối trả lời và không suy luận gì thêm nếu ngữ cảnh không có câu trả lời.

Hãy trả lời câu hỏi dựa trên ngữ cảnh:
### Ngữ cảnh :
{context}

### Câu hỏi :
{query}

### Trả lời :"""
        
        return prompt
    
    def _format_few_shot_examples(self, intent: str) -> str:
        """Format few-shot examples for intent"""
        examples = self.few_shot_examples.get(intent, [])
        
        if not examples:
            return ""
        
        formatted = "VÍ DỤ THAM KHẢO:\n"
        for i, ex in enumerate(examples, 1):
            formatted += f"\nVí dụ {i}:\n"
            formatted += f"Tài liệu: {ex['context']}\n"
            formatted += f"Câu hỏi: {ex['question']}\n"
            formatted += f"Trả lời: {ex['answer']}\n"
        
        formatted += "\nBÂY GIỜ HÃY TRẢ LỜI CÂU HỎI SAU:\n"
        
        return formatted


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
    intent = "tuition_fee"
    
    prompt = generator.generate_rag_prompt(query, context_items, intent)
    
    print(prompt)
