import gradio as gr
import os
import json
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime

from rag_pipeline import RAGPipeline
from processors.pdf_processor import PDFProcessor
from config.config import UPLOADED_PDFS_DIR, EMBEDDINGS_INDEX_FILE, EMBEDDINGS_DATA_FILE, TEXT_CHUNKS_FILE

# --- Global RAG Pipeline Instance ---
# Khởi tạo pipeline một lần và tái sử dụng nó
rag_pipeline = RAGPipeline()

def check_data_exists():
    """Kiểm tra xem có dữ liệu nào để tải không"""
    return (EMBEDDINGS_INDEX_FILE.exists() and
            EMBEDDINGS_DATA_FILE.exists() and
            TEXT_CHUNKS_FILE.exists())

def load_existing_data():
    """Tải embeddings và text chunks hiện có nếu có"""
    try:
        if check_data_exists():
            embeddings_data = np.load(EMBEDDINGS_DATA_FILE, allow_pickle=True)
            embeddings = np.array(embeddings_data['embeddings'], dtype=np.float32)
            with open(TEXT_CHUNKS_FILE, 'r', encoding='utf-8') as f:
                text_chunks = json.load(f)
            with open(EMBEDDINGS_INDEX_FILE, 'r', encoding='utf-8') as f:
                embeddings_index = json.load(f)
            return embeddings, text_chunks, embeddings_index
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu hiện có: {e}")
    return None, None, {}

def save_data(embeddings, text_chunks, embeddings_index):
    """Lưu embeddings và text chunks vào đĩa"""
    try:
        EMBEDDINGS_DIR = EMBEDDINGS_DATA_FILE.parent
        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype=np.float32)
        np.savez_compressed(EMBEDDINGS_DATA_FILE, embeddings=embeddings)
        with open(TEXT_CHUNKS_FILE, 'w', encoding='utf-8') as f:
            json.dump(text_chunks, f, ensure_ascii=False, indent=2)
        with open(EMBEDDINGS_INDEX_FILE, 'w', encoding='utf-8') as f:
            json.dump(embeddings_index, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Lỗi khi lưu dữ liệu: {e}")
        raise

def initialize_pipeline():
    """Khởi tạo hoặc tải pipeline RAG"""
    global rag_pipeline
    if not rag_pipeline.is_initialized and check_data_exists():
        print("Đang tải dữ liệu hiện có vào pipeline...")
        try:
            embeddings, text_chunks, _ = load_existing_data()
            if embeddings is not None and text_chunks is not None:
                rag_pipeline.embeddings = embeddings
                rag_pipeline.text_chunks = text_chunks
                rag_pipeline.setup_pipeline(load_existing_embeddings=True)
                print("Pipeline đã được khởi tạo thành công với dữ liệu hiện có.")
        except Exception as e:
            print(f"Lỗi khi khởi tạo pipeline: {e}")

def add_pdf_and_process(file_obj):
    """Xử lý tệp PDF được tải lên và cập nhật pipeline RAG"""
    if file_obj is None:
        return "Vui lòng chọn một tệp PDF.", ""

    original_name = os.path.basename(file_obj.name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = original_name.replace(" ", "_")
    pdf_filename = f"{timestamp}_{safe_filename}"
    pdf_path = UPLOADED_PDFS_DIR / pdf_filename

    try:
        UPLOADED_PDFS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(file_obj.name, pdf_path)

        # Đảm bảo pipeline được khởi tạo
        if not rag_pipeline.is_initialized:
            rag_pipeline.setup_pipeline(load_existing_embeddings=True)

        # Xử lý PDF mới và thêm vào kho kiến thức
        pdf_processor = PDFProcessor(pdf_path)
        pages_and_texts = pdf_processor.process_pdf()
        text_chunks = rag_pipeline.text_processor.process_text(pages_and_texts)
        new_embeddings = rag_pipeline.embedding_manager.create_embeddings(text_chunks)

        embeddings, _, embeddings_index = load_existing_data()

        if embeddings is not None and len(embeddings) > 0:
            combined_embeddings = np.vstack([embeddings, new_embeddings])
            rag_pipeline.text_chunks.extend(text_chunks)
        else:
            combined_embeddings = new_embeddings
            rag_pipeline.text_chunks = text_chunks

        rag_pipeline.embeddings = combined_embeddings
        rag_pipeline.retrieval_system.update_embeddings(rag_pipeline.embeddings, rag_pipeline.text_chunks)

        embeddings_index[str(pdf_path)] = {'original_name': original_name, 'timestamp': timestamp}
        save_data(rag_pipeline.embeddings, rag_pipeline.text_chunks, embeddings_index)

        return f"PDF '{original_name}' đã được xử lý thành công!", get_loaded_documents_list()

    except Exception as e:
        if pdf_path.exists():
            pdf_path.unlink()
        return f"Lỗi khi xử lý PDF: {str(e)}", get_loaded_documents_list()

def respond(message, chat_history):
    """Tạo phản hồi cho tin nhắn của người dùng"""
    if not rag_pipeline.is_initialized:
        return "", chat_history + [[message, "Hệ thống chưa sẵn sàng. Vui lòng tải lên tài liệu PDF."]]
    
    try:
        answer = rag_pipeline.ask(message)
        chat_history.append((message, answer))
        return "", chat_history
    except Exception as e:
        error_message = f"Lỗi: {str(e)}"
        chat_history.append((message, error_message))
        return "", chat_history

def get_loaded_documents_list():
    """Lấy danh sách các tài liệu đã được tải và xử lý"""
    _, _, embeddings_index = load_existing_data()
    if not embeddings_index:
        return "Chưa có tài liệu nào được tải lên."
    
    doc_list = []
    for info in embeddings_index.values():
        doc_list.append(f"📄 {info['original_name']} (Thêm vào: {info['timestamp']})")
    return "\n".join(doc_list)

# --- Giao diện Gradio ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# RAG Chatbot với Gradio")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Tải lên tài liệu PDF")
            file_output = gr.Textbox(label="Trạng thái xử lý", interactive=False)
            pdf_upload = gr.File(label="Tải lên PDF", file_types=[".pdf"])
            
            gr.Markdown("### Tài liệu đã tải")
            loaded_docs = gr.Textbox(value=get_loaded_documents_list, label="Danh sách tài liệu", interactive=False, lines=10)
            
            pdf_upload.upload(add_pdf_and_process, inputs=[pdf_upload], outputs=[file_output, loaded_docs])

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat")
            msg = gr.Textbox(label="Nhập câu hỏi của bạn", placeholder="Hỏi bất cứ điều gì về tài liệu của bạn...")
            clear = gr.ClearButton([msg, chatbot])

            msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    initialize_pipeline()
    demo.launch(share=True)