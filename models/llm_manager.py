# llm_manager.py
"""
LLM management module for the RAG pipeline (supports both Gemini API and local PEFT models)
"""
import json
import requests
import torch
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available
from peft import PeftModel, PeftConfig

from config.config import (
    LLM_DEVICE, USE_QUANTIZATION,
    DEFAULT_TEMPERATURE, DEFAULT_MAX_NEW_TOKENS,
    USE_REMOTE, API_KEY, REMOTE_MODEL_NAME,
    LLM_MODEL_PATH,
)
from utils.utils import get_gpu_memory_gb, recommend_model_config, get_model_mem_size


class LLMManager:
    """Handles LLM loading, configuration, and text generation"""

    def __init__(self, model_path: str = str(LLM_MODEL_PATH), device: str = LLM_DEVICE):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.use_quantization = USE_QUANTIZATION
        self.use_remote = USE_REMOTE

    # ---------------- Local (HF) helpers ----------------

    def _setup_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Setup 4-bit quantization to save VRAM"""
        if not self.use_quantization:
            return None
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

    def load_model(self):
        """Load model - Gemini API hoặc local model"""
        if self.use_remote:
            # Gemini API
            print(f"[INFO] Using Gemini API: {REMOTE_MODEL_NAME}")
            self.model = None
            self.tokenizer = None
            return
        
        # Load local model
        from pathlib import Path
        model_path = Path(self.model_path)
        
        if not model_path.exists() or not any(model_path.iterdir()):
            raise RuntimeError(f"Model not found in {model_path}. Please copy your model there.")
        
        print(f"[INFO] Loading model from: {model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True
        )
        
        # Setup model kwargs
        model_kwargs = {
            "pretrained_model_name_or_path": str(model_path),
            "torch_dtype": torch.float16,
            "device_map": "auto" if self.device == "cuda" else None,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        
        # Add quantization
        if self.use_quantization:
            print("[INFO] Using 4-bit quantization")
            model_kwargs["quantization_config"] = self._setup_quantization_config()
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        
        print(f"[INFO] ✅ Model loaded on {self.device}")
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
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.5,  # HIGHER: prevent repetition and hallucination
                no_repeat_ngram_size=4,  # Prevent 4-gram repetition
                top_p=0.9,  # Nucleus sampling for more focused output
                top_k=50,   # Limit vocabulary for more deterministic output
                **kwargs,
            )

        # Decode with skip_special_tokens=True to remove <|im_start|>, <|im_end|>, etc.
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
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
        """Universal RAG prompt for all models and languages"""
        # Format context
        context_parts = []
        for i, item in enumerate(context_items, 1):
            context_parts.append(f"Context {i} (Page {item.get('page_number', 'N/A')}):\n{item['sentence_chunk']}")
        context = "\n\n".join(context_parts)
        
        # Format chat history if provided
        history_text = ""
        if chat_history and len(chat_history) > 0:
            history_parts = []
            for msg in chat_history[-6:]:  # Only keep last 6 messages
                if msg.get('role') == 'user':
                    history_parts.append(f"Người dùng: {msg.get('content', '')}")
                elif msg.get('role') == 'assistant':
                    history_parts.append(f"Trợ lý: {msg.get('content', '')}")
            if history_parts:
                history_text = f"\n\nLỊCH SỬ CUỘC TRÒ CHUYỆN:\n" + "\n".join(history_parts) + "\n"
        
        # Strict prompt: prevent hallucination, use only document info
        base_prompt = f"""Bạn là trợ lý AI của trường Đại học FPT. Trả lời câu hỏi DỰA HOÀN TOÀN vào tài liệu.

QUY TẮC BẮT BUỘC:
1. CHỈ trích dẫn CHÍNH XÁC thông tin có trong tài liệu
2. TUYỆT ĐỐI KHÔNG bịa số liệu, ngày tháng, link, hoặc thông tin không có
3. Nếu không tìm thấy thông tin: "Tài liệu không đề cập đến thông tin này"
4. Trả lời NGẮN (2-3 câu), CHÍNH XÁC, KHÔNG giải thích dài dòng

TÀI LIỆU:
{context}{history_text}

CÂU HỎI: {query}

TRẢ LỜI (ngắn gọn, chính xác):"""
        
        return base_prompt

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
            # Remove prompt from output (special tokens already removed by tokenizer)
            answer = output_text.replace(prompt, "").strip()
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
