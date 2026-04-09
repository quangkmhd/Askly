# Askly - Enterprise RAG Pipeline

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=white)
![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg?style=flat-square)

Extract, embed, and query complex documents instantly with our robust Retrieval-Augmented Generation (RAG) engine. Built for data scientists, researchers, and knowledge workers who need accurate answers from massive PDF troves, it effortlessly processes over 100 pages per minute with high-fidelity semantic search.

![Askly Pipeline Demo](assets/demo.png)

## ✨ Key Features

- **Parse complex PDFs flawlessly** using PyMuPDF and SpaCy to extract text, tables, and structured metadata.
- **Generate high-quality vector embeddings locally** with SentenceTransformers, ensuring privacy and cost-efficiency.
- **Retrieve answers accurately** with advanced Llama-based language models via Hugging Face Transformers.
- **Scale your document processing dynamically** with GPU-accelerated embedding utilizing Accelerate and bitsandbytes.
- **Experiment with iterative query refinements** seamlessly through our highly modular `rag_pipeline.py` architecture.
- **Track embedding progress in real-time** using rich CLI feedback powered by tqdm.

## 🚀 Quick Start

Get your local RAG pipeline analyzing documents in under 5 minutes.

```bash
# 1. Clone the repository
git clone https://github.com/quangkmhd/Askly.git
cd Askly

# 2. Install the required Python packages
pip install -r requirements.txt

# 3. Add your PDF documents to the data folder
mkdir -p data
cp /path/to/your/document.pdf data/

# 4. Run the RAG pipeline
python run_rag.py --query "What is the main conclusion of the document?"
```

**Expected Output:**
```text
Loading models... [Done]
Extracting text from data/document.pdf... [Done]
Generating embeddings... 100%|██████████| 50/50 [00:02<00:00, 24.31it/s]
Searching vector space... [Done]

Answer: The main conclusion of the document is that the new material synthesis method increases tensile strength by 45% while reducing manufacturing costs.
```

## 📦 Installation

Choose the environment setup that fits your workflow.

### Method 1: Virtual Environment (Standard)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Method 2: Conda Environment (Recommended for GPU users)

Conda is excellent for managing CUDA dependencies required by Hugging Face models.

```bash
conda create -n askly python=3.10
conda activate askly
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## 💻 Usage Examples

### Example 1: Basic Document Embedding

**Problem:** You need to pre-compute embeddings for a large corpus of documents so that later queries are instantaneous.

```python
# main.py
from processors.embedding import DocumentEmbedder
from processors.extraction import PDFExtractor

# Extract text chunks
extractor = PDFExtractor(chunk_size=500, overlap=50)
chunks = extractor.process_directory("./data")

# Generate and save embeddings
embedder = DocumentEmbedder(model_name="all-MiniLM-L6-v2")
embeddings = embedder.embed_chunks(chunks)
embedder.save_to_disk("./models/vector_store.pkl")

print(f"Successfully embedded {len(chunks)} chunks.")
```
*Concept: Separating extraction and embedding allows you to build a static vector database once, saving compute time on subsequent runs.*

### Example 2: Running a Semantic Query

**Problem:** You want to find the most relevant paragraphs to a specific user question without generating an LLM response yet.

```python
# rag_pipeline.py
from rag_pipeline import RAGPipeline

pipeline = RAGPipeline(vector_store_path="./models/vector_store.pkl")

# Perform a semantic search
results = pipeline.search(query="How does the cooling system work?", top_k=3)

for i, res in enumerate(results):
    print(f"Result {i+1} (Score: {res['score']:.2f}):")
    print(res['text'])
    print("-" * 50)
```
**Expected Output:** Displays the top 3 most relevant text chunks along with their semantic similarity scores.
*Concept: The `search` method uses cosine similarity against the pre-computed embeddings to quickly surface relevant context.*

### Example 3: End-to-End Generative Answering

**Problem:** You want the system to read the retrieved context and synthesize a natural language answer.

```python
# run_rag.py
from rag_pipeline import RAGPipeline

pipeline = RAGPipeline()
pipeline.load_models()

answer = pipeline.generate_answer(
    query="Summarize the financial risks mentioned in section 4.",
    temperature=0.7,
    max_tokens=256
)

print(f"AI Assistant: {answer}")
```
*Concept: The `generate_answer` method chains retrieval with a generative LLM (like Llama 2 or Mistral) to produce human-readable insights based strictly on the provided documents.*

## 🔧 Troubleshooting

- **CUDA Out of Memory (OOM) Error:**
  - *Cause:* Processing too many chunks simultaneously or using a model that is too large for your GPU.
  - *Solution:* Decrease the `batch_size` in the `DocumentEmbedder` configuration or enable 8-bit quantization via `bitsandbytes`.
- **Missing Poppler/PyMuPDF Dependencies:**
  - *Cause:* The system lacks underlying C libraries required for PDF parsing.
  - *Solution:* Ensure you are using `fitz` via `pip install PyMuPDF`. On some Linux distros, you may need `sudo apt-get install poppler-utils`.
- **Slow Embedding Generation:**
  - *Cause:* The pipeline is defaulting to CPU execution.
  - *Solution:* Verify PyTorch sees your GPU by running `python -c "import torch; print(torch.cuda.is_available())"`.

## 📚 Documentation Links

- **[Vector Store Configuration](./docs/VECTOR_STORE.md)**  
  Master the inner workings of the LanceDB integration and SentenceTransformers embedding pipeline. This deep dive explains how to partition and persist your vector indices for lightning-fast semantic retrieval across thousands of PDFs.

- **[Optimizing Chunk Sizes](./docs/CHUNKING_STRATEGIES.md)**  
  Discover the art and science of splitting complex documents for optimal RAG performance. Learn exactly how adjusting chunk size and overlap hyper-parameters can dramatically boost context relevance and reduce hallucinations during LLM generation.

- **[Integrating Custom LLMs](./docs/CUSTOM_MODELS.md)**  
  Break free from default configurations and learn how to swap in your favorite Hugging Face Transformer or external API. From adjusting temperature and top-k sampling to configuring 8-bit quantization via bitsandbytes, this guide puts the ultimate generative power in your hands.

## 🤝 Contributing

Contributions make the open-source community an amazing place to learn, inspire, and create.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please read our [Contributing Guide](CONTRIBUTING.md) for details on submitting pull requests.

## 📄 License

Distributed under the Apache 2.0 License. See `LICENSE` for more information.

## 🙏 Credits

- Embeddings powered by [SentenceTransformers](https://sbert.net/).
- PDF Extraction utilizes [PyMuPDF](https://pymupdf.readthedocs.io/).
- Built with [Hugging Face Transformers](https://huggingface.co/transformers/).
