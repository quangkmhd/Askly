# RAG Pipeline - Code Structure

## Project Structure

```
simple-local-rag/
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py              # Configuration settings
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── pdf_processor.py       # PDF downloading and text extraction
│   │   └── text_processor.py      # Text processing and chunking
│   ├── models/
│   │   ├── __init__.py
│   │   ├── embedding_manager.py   # Embedding creation and management
│   │   ├── retrieval_system.py    # Semantic search and retrieval
│   │   └── llm_manager.py         # LLM loading and text generation
│   ├── utils/
│   │   ├── __init__.py
│   │   └── utils.py               # Utility functions
│   └── rag_pipeline.py            # Main pipeline orchestrator
├── data/                          # PDF files and raw data
├── models/                        # Downloaded models
├── outputs/                       # Generated embeddings and outputs
├── main.py                        # Command-line interface
├── run_rag.py                     # Simple runner script
└── README_STRUCTURE.md            # This file
```

## Module Descriptions

### Core Pipeline
- **`rag_pipeline.py`**: Main orchestrator that coordinates all components
- **`main.py`**: Command-line interface with multiple run modes
- **`run_rag.py`**: Simple script for quick testing

### Configuration
- **`config/config.py`**: All configuration settings, paths, and parameters

### Data Processing
- **`processors/pdf_processor.py`**: Downloads PDFs and extracts text
- **`processors/text_processor.py`**: Processes text, splits sentences, creates chunks

### AI Models
- **`models/embedding_manager.py`**: Creates and manages text embeddings
- **`models/retrieval_system.py`**: Performs semantic search and retrieval
- **`models/llm_manager.py`**: Loads and manages the language model

### Utilities
- **`utils/utils.py`**: Helper functions for text processing, model management, etc.

## Usage Examples

### Command Line Interface
```bash
# Interactive mode
python main.py --mode interactive

# Demo mode with predefined questions
python main.py --mode demo

# Single question
python main.py --mode single --question "What are macronutrients?"

# Custom settings
python main.py --mode single --question "What is protein?" --temperature 0.5 --max-tokens 512
```

### Simple Runner
```bash
# Quick start
python run_rag.py
```

### Programmatic Usage
```python
from src.rag_pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline()

# Setup (downloads PDF, creates embeddings, loads models)
pipeline.setup_pipeline()

# Ask questions
answer = pipeline.ask("What are the macronutrients?")
print(answer)

# Search without generation
results = pipeline.search("protein sources")
```

## Key Features

1. **Modular Design**: Each component is separate and can be used independently
2. **Configuration Management**: All settings centralized in config.py
3. **Error Handling**: Comprehensive error handling throughout
4. **Multiple Interfaces**: CLI, programmatic, and interactive modes
5. **GPU Support**: Automatic GPU detection and model optimization
6. **Caching**: Saves embeddings to avoid recomputation
7. **Extensible**: Easy to add new models or processing steps

## Dependencies

The code requires the same dependencies as the original notebook:
- PyMuPDF (fitz)
- sentence-transformers
- transformers
- torch
- pandas
- numpy
- spacy
- tqdm
- requests

## File Organization

Each module has a specific responsibility:
- **PDF Processing**: Downloads and extracts text from PDFs
- **Text Processing**: Splits text into sentences and chunks
- **Embedding Management**: Creates and stores text embeddings
- **Retrieval**: Finds relevant documents for queries
- **LLM Management**: Generates answers using language models
- **Pipeline**: Orchestrates all components together
