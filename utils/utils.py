"""
Utility functions for the RAG pipeline
"""
import textwrap
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from time import perf_counter as timer


def print_wrapped(text: str, wrap_length: int = 80) -> None:
    """Print text with word wrapping"""
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)


def split_list(input_list: List[str], slice_size: int = 10) -> List[List[str]]:
    """
    Split a list into chunks of specified size
    e.g. [20] -> [10, 10] or [25] -> [10, 10, 5]
    """
    return [input_list[i:i+slice_size] for i in range(0, len(input_list), slice_size)]


def get_model_num_params(model: torch.nn.Module) -> int:
    """Get the number of parameters in a model"""
    return sum([param.numel() for param in model.parameters()])


def get_model_mem_size(model: torch.nn.Module) -> Dict[str, Any]:
    """Get the memory size of a model in bytes, MB, and GB"""
    # Get model parameters and buffer sizes
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])

    # Calculate model sizes
    model_mem_bytes = mem_params + mem_buffers
    model_mem_mb = model_mem_bytes / (1024**2)
    model_mem_gb = model_mem_bytes / (1024**3) 

    return {
        "model_mem_bytes": model_mem_bytes,
        "model_mem_mb": round(model_mem_mb, 2), 
        "model_mem_gb": round(model_mem_gb, 2)
    }


def dot_product(vector1: torch.Tensor, vector2: torch.Tensor) -> torch.Tensor:
    """Calculate dot product between two vectors"""
    return torch.dot(vector1, vector2)


def cosine_similarity(vector1: torch.Tensor, vector2: torch.Tensor) -> torch.Tensor:
    """Calculate cosine similarity between two vectors"""
    dot_product_val = torch.dot(vector1, vector2)

    # Get Euclidean/L2 norm
    norm_vector1 = torch.sqrt(torch.sum(vector1**2))
    norm_vector2 = torch.sqrt(torch.sum(vector2**2))

    return dot_product_val / (norm_vector1 * norm_vector2)


def get_gpu_memory_gb() -> int:
    """Get available GPU memory in GB"""
    if torch.cuda.is_available():
        gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_gb = round(gpu_memory_bytes / (2**30))
        return gpu_memory_gb
    return 0


def recommend_model_config(gpu_memory_gb: int) -> Tuple[str, bool]:
    """
    Recommend model configuration based on GPU memory
    Returns (model_id, use_quantization)
    """
    if gpu_memory_gb < 5.1:
        return "google/gemma-2b-it", True
    elif gpu_memory_gb < 8.1:
        return "google/gemma-2b-it", True
    elif gpu_memory_gb < 19.0:
        return "google/gemma-2b-it", False
    else:
        return "google/gemma-7b-it", False


def format_time(seconds: float) -> str:
    """Format time in a human-readable way"""
    if seconds < 1:
        return f"{seconds*1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.2f}s"


def clean_text(text: str) -> str:
    """Clean and format text"""
    import re
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Fix spacing after periods
    text = re.sub(r'\.([A-Z])', r'. \1', text)
    return text


def print_progress_bar(iteration: int, total: int, prefix: str = '', suffix: str = '', 
                      decimals: int = 1, length: int = 50, fill: str = '█') -> None:
    """Print a progress bar"""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()
