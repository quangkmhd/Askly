"""
LLM management module for the RAG pipeline
"""
import torch
from typing import List, Dict, Any, Optional, Tuple
import requests
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available

from config.config import (
    LLM_MODEL_ID, LLM_DEVICE, LLM_TORCH_DTYPE, USE_QUANTIZATION,
    USE_FLASH_ATTENTION, DEFAULT_TEMPERATURE, DEFAULT_MAX_NEW_TOKENS,
    USE_REMOTE, BASE_URL, API_KEY, REMOTE_MODEL_NAME
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
        
    def _setup_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Setup quantization configuration"""
        if not self.use_quantization:
            return None
        
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
    
    def _setup_attention_implementation(self) -> str:
        """Setup attention implementation based on hardware capabilities"""
        if USE_FLASH_ATTENTION and torch.cuda.is_available():
            if (is_flash_attn_2_available() and 
                torch.cuda.get_device_capability(0)[0] >= 8):
                return "flash_attention_2"
        
        return "sdpa"  # scaled dot product attention
    
    def _auto_configure_model(self) -> Tuple[str, bool]:
        """Auto-configure model based on available GPU memory"""
        gpu_memory_gb = get_gpu_memory_gb()
        if gpu_memory_gb == 0:
            print("[WARNING] No GPU detected, using CPU")
            return "google/gemma-2b-it", True
        
        model_id, use_quantization = recommend_model_config(gpu_memory_gb)
        print(f"[INFO] GPU memory: {gpu_memory_gb}GB | Recommended model: {model_id}")
        print(f"[INFO] Use quantization: {use_quantization}")
        
        return model_id, use_quantization
    
    def load_model(self, auto_configure: bool = True):
        """Load the LLM model/tokenizer or set up remote API usage"""
        if self.use_remote:
            # Remote inference does not require local model
            print(f"[INFO] Using remote model '{REMOTE_MODEL_NAME}' via {BASE_URL}")
            self.model_id = REMOTE_MODEL_NAME
            self.tokenizer = None  # not needed
            self.model = None
            return

        """Load the LLM model and tokenizer"""
        # Local model path (legacy)
        if auto_configure:
            self.model_id, self.use_quantization = self._auto_configure_model()
        
        print(f"[INFO] Loading LLM model: {self.model_id}")
        
        # Setup configurations
        quantization_config = self._setup_quantization_config()
        self.attn_implementation = self._setup_attention_implementation()
        
        print(f"[INFO] Using attention implementation: {self.attn_implementation}")
        print(f"[INFO] Using quantization: {self.use_quantization}")
        
        # Load tokenizer
        print("[INFO] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.model_id
        )
        
        # Load model
        print("[INFO] Loading model...")
        model_kwargs = {
            "pretrained_model_name_or_path": self.model_id,
            "torch_dtype": getattr(torch, LLM_TORCH_DTYPE),
            "low_cpu_mem_usage": False,
            "attn_implementation": self.attn_implementation
        }
        
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        
        self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        
        # Move to device if not using quantization
        if not self.use_quantization and self.device != "cpu":
            self.model.to(self.device)
        
        print(f"[INFO] Model loaded successfully on {self.device}")
        
        # Print model info
        self._print_model_info()
    
    def _print_model_info(self):
        """Print model information"""
        if self.model is None:
            return
        
        num_params = sum([param.numel() for param in self.model.parameters()])
        mem_info = get_model_mem_size(self.model)
        
        print(f"[INFO] Model parameters: {num_params:,}")
        print(f"[INFO] Model memory: {mem_info['model_mem_gb']:.2f} GB")
        print(f"[INFO] Model device: {next(self.model.parameters()).device}")
    
    def _remote_generate(self, prompt: str, temperature: float, max_new_tokens: int, stream: bool = False) -> str:
        """Generate text by calling the remote API"""
        payload = {
            "model": REMOTE_MODEL_NAME,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "stream": stream
        }
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        if stream:
            return self._handle_stream_response(f"{BASE_URL}/v1/chat/completions", headers, payload)
        else:
            resp = requests.post(f"{BASE_URL}/v1/chat/completions", headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
    
    def _handle_stream_response(self, url: str, headers: dict, data: dict) -> str:
        """Handle streaming response from remote API"""
        import json
        
        response = requests.post(url, headers=headers, json=data, stream=True)
        response.raise_for_status()
        
        full_content = ""
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    line = line[6:]  # Remove 'data: ' prefix
                    
                    if line.strip() == '[DONE]':
                        break
                    
                    try:
                        chunk_data = json.loads(line)
                        if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                            delta = chunk_data['choices'][0].get('delta', {})
                            if 'content' in delta:
                                content = delta['content']
                                full_content += content
                                print(content, end='', flush=True)
                    except json.JSONDecodeError:
                        continue
        
        return full_content

    def generate_text(self, prompt: str, temperature: float = DEFAULT_TEMPERATURE,
                     max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                     do_sample: bool = True, stream: bool = False, **kwargs) -> str:
        """Generate text either locally or via remote API"""
        if self.use_remote:
            return self._remote_generate(prompt, temperature, max_new_tokens, stream)

        # Local generation path
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")

        # Tokenize input
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **input_ids,
                temperature=temperature,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                **kwargs
            )

        # Decode output
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        return output_text
    
    def generate_with_template(self, query: str, temperature: float = DEFAULT_TEMPERATURE,
                             max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> str:
        """Generate text using the model's chat template"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")
        
        # Create dialogue template
        dialogue_template = [
            {"role": "user", "content": query}
        ]
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            conversation=dialogue_template,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate
        output_text = self.generate_text(
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )
        
        return output_text
    
    def format_rag_prompt(self, query: str, context_items: List[Dict[str, Any]]) -> str:
        """Format a RAG prompt with context items"""
        if self.use_remote:
            # Build detailed prompt with better context formatting
            context_parts = []
            for i, item in enumerate(context_items, 1):
                context_parts.append(f"Context {i} (Page {item.get('page_number', 'N/A')}):\n{item['sentence_chunk']}")
            
            context = "\n\n".join(context_parts)
            base_prompt = f"""You are a helpful assistant that answers questions based on the provided context from a Human Nutrition textbook.

CONTEXT:
{context}

INSTRUCTIONS:
- Answer the user's question based ONLY on the information provided in the context above
- If the context contains fragmented or incomplete information, piece it together logically
- If you cannot find the specific information requested, say so clearly
- Be precise and detailed in your response
- If the user asks for a list or table of contents, provide the exact structure as shown in the context

USER QUERY: {query}

ANSWER:"""
            return base_prompt

        if self.tokenizer is None:
            raise ValueError("Tokenizer must be loaded first")
        
        # Create context string
        context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])
        
        # Base prompt template
        base_prompt = f"""Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
Use the following examples as reference for the ideal answer style.

Example 1:
Query: What are the fat-soluble vitamins?
Answer: The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K. These vitamins are absorbed along with fats in the diet and can be stored in the body's fatty tissue and liver for later use. Vitamin A is important for vision, immune function, and skin health. Vitamin D plays a critical role in calcium absorption and bone health. Vitamin E acts as an antioxidant, protecting cells from damage. Vitamin K is essential for blood clotting and bone metabolism.

Example 2:
Query: What are the causes of type 2 diabetes?
Answer: Type 2 diabetes is often associated with overnutrition, particularly the overconsumption of calories leading to obesity. Factors include a diet high in refined sugars and saturated fats, which can lead to insulin resistance, a condition where the body's cells do not respond effectively to insulin. Over time, the pancreas cannot produce enough insulin to manage blood sugar levels, resulting in type 2 diabetes. Additionally, excessive caloric intake without sufficient physical activity exacerbates the risk by promoting weight gain and fat accumulation, particularly around the abdomen, further contributing to insulin resistance.

Example 3:
Query: What is the importance of hydration for physical performance?
Answer: Hydration is crucial for physical performance because water plays key roles in maintaining blood volume, regulating body temperature, and ensuring the transport of nutrients and oxygen to cells. Adequate hydration is essential for optimal muscle function, endurance, and recovery. Dehydration can lead to decreased performance, fatigue, and increased risk of heat-related illnesses, such as heat stroke. Drinking sufficient water before, during, and after exercise helps ensure peak physical performance and recovery.

Now use the following context items to answer the user query:
{context}

Relevant passages: <extract relevant passages from the context here>
User query: {query}
Answer:"""
        
        # Create dialogue template
        dialogue_template = [
            {"role": "user", "content": base_prompt}
        ]
        
        # Apply chat template
        try:
            prompt = self.tokenizer.apply_chat_template(
                conversation=dialogue_template,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # If apply_chat_template returns None or empty string, use base_prompt
            if not prompt:
                prompt = base_prompt
                
        except (AttributeError, NotImplementedError):
            # Fallback to base_prompt if tokenizer doesn't support chat templates
            prompt = base_prompt
        
        return prompt
    
    def generate_rag_response(self, query: str, context_items: List[Dict[str, Any]],
                            temperature: float = DEFAULT_TEMPERATURE,
                            max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                            format_answer: bool = True, stream: bool = False) -> str:
        """Generate a RAG response with context"""
        # Format prompt
        prompt = self.format_rag_prompt(query, context_items)
        
        # Generate response
        output_text = self.generate_text(
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            stream=stream
        )
        
        # Format answer if requested
        if format_answer:
            # Remove prompt and special tokens
            answer = output_text.replace(prompt, "").replace("<bos>", "").replace("<eos>", "").strip()
            return answer
        
        return output_text
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if self.model is None:
            return {}
        
        return {
            "model_id": self.model_id,
            "device": str(next(self.model.parameters()).device),
            "dtype": str(next(self.model.parameters()).dtype),
            "num_parameters": sum([param.numel() for param in self.model.parameters()]),
            "memory_info": get_model_mem_size(self.model),
            "attention_implementation": self.attn_implementation,
            "quantization_enabled": self.use_quantization
        }
