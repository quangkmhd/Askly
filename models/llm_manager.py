# llm_manager.py
"""
LLM management module for the RAG pipeline (Gemini REST API version)
"""
import json
import requests
import torch
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available

from config.config import (
    LLM_MODEL_ID, LLM_DEVICE, LLM_TORCH_DTYPE, USE_QUANTIZATION,
    USE_FLASH_ATTENTION, DEFAULT_TEMPERATURE, DEFAULT_MAX_NEW_TOKENS,
    USE_REMOTE, BASE_URL, API_KEY, REMOTE_MODEL_NAME,
)
from utils.utils import get_gpu_memory_gb, recommend_model_config, get_model_mem_size


class LLMManager:
    """Handles LLM loading, configuration, and text generation"""

    def __init__(self, model_id: str = LLM_MODEL_ID, device: str = LLM_DEVICE):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.tokenizer = None
        self.use_quantization = USE_QUANTIZATION
        self.attn_implementation = None
        self.use_remote = USE_REMOTE

    # ---------------- Local (HF) helpers ----------------

    def _setup_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        if not self.use_quantization:
            return None
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    def _setup_attention_implementation(self) -> str:
        if USE_FLASH_ATTENTION and torch.cuda.is_available():
            if is_flash_attn_2_available() and torch.cuda.get_device_capability(0)[0] >= 8:
                return "flash_attention_2"
        return "sdpa"

    def _auto_configure_model(self) -> Tuple[str, bool]:
        gpu_memory_gb = get_gpu_memory_gb()
        if gpu_memory_gb == 0:
            print("[WARNING] No GPU detected, using CPU")
            return "google/gemma-2b-it", True
        model_id, use_quantization = recommend_model_config(gpu_memory_gb)
        print(f"[INFO] GPU memory: {gpu_memory_gb}GB | Recommended model: {model_id}")
        print(f"[INFO] Use quantization: {use_quantization}")
        return model_id, use_quantization

    # ---------------- Load (remote vs local) ----------------

    def load_model(self, auto_configure: bool = True):
        """Either set up remote (Gemini) or load local HF model"""
        if self.use_remote:
            # Remote inference via Google Generative Language API (Gemini)
            print(f"[INFO] Using remote Gemini model '{REMOTE_MODEL_NAME}' via Google Generative Language API")
            self.model_id = REMOTE_MODEL_NAME
            self.tokenizer = None
            self.model = None
            return

        # Local Hugging Face model (fallback)
        if auto_configure:
            self.model_id, self.use_quantization = self._auto_configure_model()

        print(f"[INFO] Loading LLM model: {self.model_id}")
        self.attn_implementation = self._setup_attention_implementation()
        print(f"[INFO] Using attention implementation: {self.attn_implementation}")
        print(f"[INFO] Using quantization: {self.use_quantization}")

        print("[INFO] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.model_id)

        print("[INFO] Loading model...")
        model_kwargs = {
            "pretrained_model_name_or_path": self.model_id,
            "torch_dtype": getattr(torch, LLM_TORCH_DTYPE),
            "low_cpu_mem_usage": False,
            "attn_implementation": self.attn_implementation,
        }
        qcfg = self._setup_quantization_config()
        if qcfg is not None:
            model_kwargs["quantization_config"] = qcfg

        self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

        if not self.use_quantization and self.device != "cpu":
            self.model.to(self.device)

        print(f"[INFO] Model loaded successfully on {self.device}")
        self._print_model_info()

    def _print_model_info(self):
        if self.model is None:
            return
        num_params = sum(p.numel() for p in self.model.parameters())
        mem_info = get_model_mem_size(self.model)
        print(f"[INFO] Model parameters: {num_params:,}")
        print(f"[INFO] Model memory: {mem_info['model_mem_gb']:.2f} GB")
        print(f"[INFO] Model device: {next(self.model.parameters()).device}")

    # ---------------- Gemini REST (remote) ----------------

    def _gemini_generate_nonstream(self, prompt: str, temperature: float, max_new_tokens: int) -> str:
        if not API_KEY:
            raise RuntimeError("Missing API_KEY for Gemini")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{REMOTE_MODEL_NAME}:generateContent"
        headers = {"Content-Type": "application/json", "X-goog-api-key": API_KEY}
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": float(temperature), "maxOutputTokens": int(max_new_tokens)},
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        # Extract text from candidates[0].content.parts[*].text
        try:
            candidates = data.get("candidates", [])
            if not candidates:
                return json.dumps(data, ensure_ascii=False)
            parts = candidates[0].get("content", {}).get("parts", [])
            txts = [p.get("text", "") for p in parts if isinstance(p, dict)]
            return "".join(txts).strip() or json.dumps(data, ensure_ascii=False)
        except Exception:
            return json.dumps(data, ensure_ascii=False)

    def _gemini_generate_stream(self, prompt: str, temperature: float, max_new_tokens: int) -> str:
        """Stream via SSE: POST ...:streamGenerateContent?alt=sse"""
        if not API_KEY:
            raise RuntimeError("Missing API_KEY for Gemini")

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{REMOTE_MODEL_NAME}:streamGenerateContent?alt=sse"
        )
        headers = {"Content-Type": "application/json", "X-goog-api-key": API_KEY}
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": float(temperature), "maxOutputTokens": int(max_new_tokens)},
        }

        resp = requests.post(url, headers=headers, json=payload, stream=True)
        resp.raise_for_status()

        full = ""
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8")
            # SSE lines typically start with: "data: {...}" or "data: [DONE]"
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                cands = chunk.get("candidates", [])
                if not cands:
                    continue
                parts = cands[0].get("content", {}).get("parts", [])
                for p in parts:
                    piece = p.get("text", "")
                    if piece:
                        full += piece
                        print(piece, end="", flush=True)  # stream ra terminal
            except Exception:
                continue
        return full.strip()

    def _remote_generate(self, prompt: str, temperature: float, max_new_tokens: int, stream: bool = False) -> str:
        """Dispatch to Gemini REST (stream or non-stream)"""
        if stream:
            return self._gemini_generate_stream(prompt, temperature, max_new_tokens)
        return self._gemini_generate_nonstream(prompt, temperature, max_new_tokens)

    # ---------------- Unified generate ----------------

    def generate_text(
        self,
        prompt: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        do_sample: bool = True,
        stream: bool = False,
        **kwargs,
    ) -> str:
        if self.use_remote:
            return self._remote_generate(prompt, temperature, max_new_tokens, stream)

        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")

        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **input_ids,
                temperature=temperature,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )

        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        return output_text

    # ---------------- RAG helpers ----------------

    def generate_with_template(
        self, query: str, temperature: float = DEFAULT_TEMPERATURE, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    ) -> str:
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")

        dialogue_template = [{"role": "user", "content": query}]
        prompt = self.tokenizer.apply_chat_template(
            conversation=dialogue_template, tokenize=False, add_generation_prompt=True
        )
        if not prompt:
            prompt = query

        return self.generate_text(prompt=prompt, temperature=temperature, max_new_tokens=max_new_tokens)

    def format_rag_prompt(self, query: str, context_items: List[Dict[str, Any]], chat_history: List[Dict[str, str]] = None) -> str:
        if self.use_remote:
            context_parts = []
            for i, item in enumerate(context_items, 1):
                context_parts.append(f"Context {i} (Page {item.get('page_number', 'N/A')}):\n{item['sentence_chunk']}")
            context = "\n\n".join(context_parts)
            
            # Detect if query is in Vietnamese
            is_vietnamese = any(c.isalpha() and ord(c) > 127 for c in query)
            
            if is_vietnamese:
                # Format chat history if provided
                history_text = ""
                if chat_history and len(chat_history) > 0:
                    history_parts = []
                    for msg in chat_history[-6:]:  # Only keep last 6 messages to avoid too long context
                        if msg.get('role') == 'user':
                            history_parts.append(f"Người dùng: {msg.get('content', '')}")
                        elif msg.get('role') == 'assistant':
                            history_parts.append(f"Trợ lý: {msg.get('content', '')}")
                    if history_parts:
                        history_text = f"\n\nLỊCH SỬ CUỘC TRÒ CHUYỆN:\n" + "\n".join(history_parts) + "\n"

                base_prompt = f"""Bạn là một trợ lý thông minh và linh hoạt, có thể trả lời câu hỏi dựa trên tài liệu và kiến thức tổng quát.

NGỮ CẢNH TỪ TÀI LIỆU:
{context}{history_text}

        HƯỚNG DẪN:
        - Trả lời trực tiếp, tự nhiên như một chuyên gia am hiểu
        - ƯU TIÊN sử dụng thông tin từ tài liệu nếu có liên quan
        - Nếu tài liệu có thông tin LIÊN QUAN nhưng không đầy đủ, hãy KẾT HỢP với kiến thức chung để đưa ra câu trả lời hoàn chỉnh
        - Nếu tài liệu không có thông tin, hãy sử dụng KIẾN THỨC TỔNG QUÁT để trả lời một cách hữu ích
        - Hãy PHÂN TÍCH và TỔNG HỢP thông tin từ nhiều đoạn ngữ cảnh để đưa ra câu trả lời hoàn chỉnh
        - Nếu câu hỏi không khớp chính xác với nội dung, hãy tìm thông tin LIÊN QUAN và suy luận hợp lý
        - Có thể đưa ra lời khuyên, giải thích khái niệm, so sánh, phân tích dựa trên hiểu biết chung
        - Nếu có lịch sử cuộc trò chuyện, hãy tham khảo để duy trì ngữ cảnh
        - Trả lời bằng tiếng Việt, sử dụng từ ngữ dễ hiểu và thân thiện
        - Hãy sáng tạo và hữu ích: có thể đưa ra ví dụ, gợi ý, hoặc thông tin bổ sung
        - Hãy suy luận thông minh: ví dụ "học phí của trường" có thể liên quan đến "học phí các ngành", "chi phí đào tạo", v.v.
        - Chỉ nói "Tôi không biết" khi thực sự không thể trả lời được

CÂU HỎI: {query}

TRẢ LỜI:"""
            else:
                # Format chat history if provided
                history_text = ""
                if chat_history and len(chat_history) > 0:
                    history_parts = []
                    for msg in chat_history[-6:]:  # Only keep last 6 messages to avoid too long context
                        if msg.get('role') == 'user':
                            history_parts.append(f"User: {msg.get('content', '')}")
                        elif msg.get('role') == 'assistant':
                            history_parts.append(f"Assistant: {msg.get('content', '')}")
                    if history_parts:
                        history_text = f"\n\nCONVERSATION HISTORY:\n" + "\n".join(history_parts) + "\n"

                base_prompt = f"""You are an intelligent and flexible assistant that can answer questions using both document context and general knowledge.

DOCUMENT CONTEXT:
{context}{history_text}

        INSTRUCTIONS:
        - Answer directly and naturally like a knowledgeable expert
        - PRIORITIZE information from the document if relevant
        - If the document has RELATED but incomplete information, COMBINE it with general knowledge for a complete answer
        - If the document lacks information, use GENERAL KNOWLEDGE to provide helpful responses
        - ANALYZE and SYNTHESIZE information from multiple context sections to provide complete answers
        - If the question doesn't match exactly, find RELATED information and make reasonable inferences
        - You can give advice, explain concepts, compare, and analyze based on common understanding
        - If there's conversation history, refer to it to maintain context
        - Answer in the same language as the user's question
        - Be creative and helpful: provide examples, suggestions, or additional information
        - Think intelligently: e.g., "school tuition" might relate to "program fees", "training costs", etc.
        - Only say "I don't know" when you truly cannot provide any useful response

USER QUERY: {query}

ANSWER:"""
            return base_prompt

        if self.tokenizer is None:
            raise ValueError("Tokenizer must be loaded first")

        context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])
        base_prompt = f"""Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.

Now use the following context items to answer the user query:
{context}

Relevant passages: <extract relevant passages from the context here>
User query: {query}
Answer:"""

        dialogue_template = [{"role": "user", "content": base_prompt}]
        try:
            prompt = self.tokenizer.apply_chat_template(
                conversation=dialogue_template, tokenize=False, add_generation_prompt=True
            )
            if not prompt:
                prompt = base_prompt
        except (AttributeError, NotImplementedError):
            prompt = base_prompt
        return prompt

    def generate_rag_response(
        self,
        query: str,
        context_items: List[Dict[str, Any]],
        temperature: float = DEFAULT_TEMPERATURE,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        format_answer: bool = True,
        stream: bool = False,
        chat_history: List[Dict[str, str]] = None,
    ) -> str:
        prompt = self.format_rag_prompt(query, context_items, chat_history)
        output_text = self.generate_text(prompt=prompt, temperature=temperature, max_new_tokens=max_new_tokens, stream=stream)

        if format_answer:
            answer = output_text.replace(prompt, "").replace("<bos>", "").replace("<eos>", "").strip()
            return answer
        return output_text

    def get_model_info(self) -> Dict[str, Any]:
        if self.model is None:
            return {}
        return {
            "model_id": self.model_id,
            "device": str(next(self.model.parameters()).device),
            "dtype": str(next(self.model.parameters()).dtype),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "memory_info": get_model_mem_size(self.model),
            "attention_implementation": self.attn_implementation,
            "quantization_enabled": self.use_quantization,
        }
