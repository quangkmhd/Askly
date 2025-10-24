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
        
        # Get intent-specific rules with natural language instructions (16 intents)
        intent_rules = {
            'tuition_fee': "\nLƯU Ý: Trích dẫn chính xác số tiền, không làm tròn. Nêu rõ học kỳ, năm học.",
            'admission': "\nLƯU Ý: Nêu rõ điều kiện xét tuyển, phương thức, thời hạn. Có cấu trúc rõ ràng.",
            'grades': "\nLƯU Ý: Phân biệt GPA, điểm xếp loại, điểm tối thiểu. Giải thích đơn giản.",
            'schedule': "\nLƯU Ý: Nêu rõ thời gian, học kỳ, tuần, tháng. Chính xác về ngày tháng.",
            'graduation': "\nLƯU Ý: Liệt kê đầy đủ điều kiện tốt nghiệp, thủ tục. Giải thích từng bước.",
            'program': "\nLƯU Ý: Mô tả chương trình, môn học, thời lượng. Phân biệt các ngành rõ ràng.",
            'research': "\nLƯU Ý: Nêu rõ quy trình đăng ký, yêu cầu, thời hạn. Giải thích các bước cụ thể.",
            'exam_rules': "\nLƯU Ý: Liệt kê nội quy chi tiết, xử lý vi phạm. Nhấn mạnh những điều cấm.",
            'dormitory': "\nLƯU Ý: Mô tả điều kiện, tiện nghi, giá cả. Hướng dẫn cách đăng ký.",
            'internship': "\nLƯU Ý: Giải thích rõ OJT, capstone, thời điểm, yêu cầu. Hướng dẫn chuẩn bị.",
            'procedures': "\nLƯU Ý: Nêu từng bước thủ tục, giấy tờ cần thiết, nơi nộp. Rất cụ thể.",
            'conduct_rules': "\nLƯU Ý: Giải thích quy tắc, hành vi được/không được phép, hậu quả vi phạm.",
            'awards': "\nLƯU Ý: Nêu rõ điều kiện, mức thưởng, thời điểm xét. Khuyến khích sinh viên.",
            'graduate_program': "\nLƯU Ý: Mô tả chương trình, thời lượng, yêu cầu đầu vào, học phí.",
            'contact': "\nLƯU Ý: Cung cấp đầy đủ SĐT, email, địa chỉ, giờ làm việc. Format rõ ràng.",
            'technology': "\nLƯU Ý: Mô tả hệ thống, cách sử dụng, troubleshooting cơ bản.",
            
            # Fallback
            'general': "\nLƯU Ý: Trả lời ngắn gọn, dễ hiểu, thân thiện."
        }
        
        special_rules = intent_rules.get(intent, intent_rules['general'])
        
        # Get short few-shot examples (1-2 only)
        examples = self._get_short_examples(intent)
        
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
- Trích dẫn chính xác thông tin từ tài liệu{special_rules}{history_context}

{examples}TÀI LIỆU:
{context}

CÂU HỎI: {query}
TRẢ LỜI:"""
        
        return prompt
    
    def _format_few_shot_examples(self, intent: str) -> str:
        """Format few-shot examples for intent (DEPRECATED - use _get_short_examples)"""
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
    
    def _get_short_examples(self, intent: str) -> str:
        """Get 1-2 concise examples for better focus"""
        if intent == 'tuition_fee':
            return """VÍ DỤ:
Tài liệu: "Học phí năm 1: 31.600.000đ/kỳ"
Hỏi: Học phí là bao nhiêu?
Đáp: Học phí năm 1 là 31.600.000đ/kỳ.

"""
        elif intent == 'admission':
            return """VÍ DỤ:
Tài liệu: "Yêu cầu: Top50 School Rank (21 điểm)"
Hỏi: Điều kiện tuyển sinh?
Đáp: Yêu cầu Top50 School Rank, tương đương 21 điểm.

"""
        elif intent == 'grades':
            return """VÍ DỤ:
Tài liệu: "GPA >= 3.0 để xếp loại Khá"
Hỏi: GPA bao nhiêu để xếp loại Khá?
Đáp: GPA >= 3.0.

"""
        return ""


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
