# Configuration & Tuning Guide: Askly

Proper configuration of Askly is critical to balancing semantic accuracy, LLM generation quality, and hardware resource constraints. This document details all configurable parameters across the pipeline.

## 1. Environment & Hardware Configuration

Askly relies heavily on Hugging Face and PyTorch. Ensure your environment is configured to leverage your hardware.

### CUDA Device Selection
If you have multiple GPUs, you can restrict Askly to a specific card to prevent Out-Of-Memory (OOM) errors affecting other workloads.
```bash
export CUDA_VISIBLE_DEVICES=0  # Uses only the first GPU
export CUDA_VISIBLE_DEVICES=0,1 # Uses first and second GPUs for data parallel tasks
```

### Hugging Face Authentication
If you are using gated models (like Llama-2 or Llama-3), you must provide a Hugging Face token.
Create a `.env` file or export the variable:
```bash
export HF_TOKEN="hf_your_long_authentication_token_here"
```

## 2. Extraction & Chunking Strategy (`config/extraction.json` or CLI args)

The way you slice your documents fundamentally determines the success of the RAG retrieval phase.

| Parameter | Description | Recommended Value | Impact |
|-----------|-------------|-------------------|--------|
| `chunk_size` | Number of characters per text block. | `500 - 1000` | Larger chunks provide more context to the LLM but dilute the semantic density for the embedder. Smaller chunks (e.g., 200) are great for factoid retrieval but poor for summarization. |
| `overlap` | Number of characters shared between chunk N and N+1. | `50 - 100` | Prevents cutting a critical sentence or concept in half. Set to roughly 10-15% of your chunk size. |

## 3. Embedding Model Selection

Configured when initializing `DocumentEmbedder(model_name="...")`.

| Model Name | Size | Speed | Quality (MTEB) | Use Case |
|------------|------|-------|----------------|----------|
| `all-MiniLM-L6-v2` | 22 MB | Extremely Fast | Good | Default. Best for local development, CPU-only execution, or massive datasets. |
| `BAAI/bge-small-en-v1.5` | 133 MB | Fast | Excellent | Highly recommended for production English RAG. |
| `BAAI/bge-m3` | 2.2 GB | Slower | State-of-the-Art | Best for multilingual support (English, Vietnamese, etc.) and complex enterprise docs. Requires GPU. |

## 4. Generative LLM Hyperparameters

When calling `pipeline.generate_answer()`, the parameters dictate the "creativity" and length of the response.

### `temperature`
- **Type:** Float (0.0 to 1.0)
- **Recommended for RAG:** `0.1` to `0.3`
- **Explanation:** Lower temperatures make the model more deterministic and focused, heavily adhering to the retrieved context. High temperatures (0.8+) make the model "creative," which in RAG applications often leads to hallucinations or ignoring the provided document data.

### `max_tokens` (or `max_new_tokens`)
- **Type:** Integer
- **Recommended:** `256` to `512`
- **Explanation:** Limits the length of the generated answer. Since RAG answers should ideally be concise and factual, setting this too high wastes compute time and increases the chance of the model rambling.

### `top_p` and `top_k`
- **Recommended `top_p`:** `0.9`
- **Recommended `top_k`:** `50`
- **Explanation:** Controls nucleus sampling. Restricts the model's vocabulary choices to only the most probable words, further reducing hallucinations.

## 5. Quantization Configuration (Advanced)

To run 7B or 13B parameter models on consumer GPUs (like an RTX 3060 with 12GB VRAM), you must configure `bitsandbytes` in the `RAGPipeline` initialization.

```python
# Inside rag_pipeline.py initialization
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Pass this config when loading the Hugging Face model
```
- **`load_in_4bit`:** Reduces memory footprint by ~75% with minimal quality loss.
- **`bnb_4bit_compute_dtype=torch.float16`:** Speeds up matrix multiplications during inference on modern GPUs.
