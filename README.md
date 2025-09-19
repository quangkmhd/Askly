# Askly: Chatbot RAG cho tài liệu PDF

Askly là một dự án chatbot sử dụng kiến trúc RAG (Retrieval-Augmented Generation) cho phép người dùng "trò chuyện" với các tài liệu PDF của mình. Bạn có thể tải lên các tệp PDF, hệ thống sẽ xử lý, lập chỉ mục nội dung và sau đó bạn có thể đặt câu hỏi bằng ngôn ngữ tự nhiên để nhận câu trả lời dựa trên thông tin trong các tài liệu đó.

Dự án cung cấp hai giao diện người dùng: một được xây dựng bằng **Streamlit** và một bằng **Gradio**.

## Tính năng chính

- **Tải lên và xử lý PDF**: Dễ dàng tải lên một hoặc nhiều tệp PDF để xây dựng cơ sở tri thức.
- **Hỏi và Đáp thông minh**: Đặt câu hỏi và nhận câu trả lời được tạo ra bởi mô hình ngôn ngữ lớn (LLM) dựa trên ngữ cảnh được truy xuất từ tài liệu của bạn.
- **Lưu trữ và Tải lại tri thức**: Các tài liệu đã xử lý (embeddings) được lưu lại, cho phép khởi động lại ứng dụng mà không cần xử lý lại từ đầu.
- **Hỗ trợ đa giao diện**: Lựa chọn giữa giao diện web Streamlit hoặc Gradio.
- **Kiến trúc module hóa**: Mã nguồn được tổ chức thành các thành phần rõ ràng (xử lý PDF, quản lý embedding, truy xuất, LLM).
- **Hỗ trợ LLM linh hoạt**: Dễ dàng chuyển đổi giữa việc sử dụng API Gemini của Google hoặc một mô hình ngôn ngữ lớn được host cục bộ.

## Kiến trúc hệ thống

Askly tuân theo một luồng RAG cổ điển:

1.  **Nạp và Phân đoạn (Ingestion & Chunking)**: Khi một tệp PDF được tải lên, nó sẽ được đọc và nội dung văn bản được chia thành các đoạn (chunks) nhỏ hơn, dễ quản lý hơn.
2.  **Tạo Embedding (Embedding Generation)**: Mỗi đoạn văn bản được chuyển đổi thành một vector số (embedding) bằng cách sử dụng một mô hình embedding (ví dụ: Universal Sentence Encoder của Google).
3.  **Lưu trữ (Storage)**: Các vector embedding và các đoạn văn bản tương ứng được lưu vào đĩa.
4.  **Truy xuất (Retrieval)**: Khi người dùng đặt câu hỏi, câu hỏi đó cũng được chuyển đổi thành một vector embedding. Hệ thống sau đó thực hiện tìm kiếm tương đồng (cosine similarity) để tìm ra các đoạn văn bản có liên quan nhất từ cơ sở tri thức.
5.  **Tăng cường và Sinh văn bản (Augmentation & Generation)**: Các đoạn văn bản liên quan nhất được chèn vào một câu lệnh (prompt) cùng với câu hỏi của người dùng. Prompt hoàn chỉnh này sau đó được gửi đến một mô hình ngôn ngữ lớn (LLM) để tạo ra câu trả lời cuối cùng.

## Yêu cầu hệ thống

- Python 3.9+
- `pip`

## Hướng dẫn cài đặt

1.  **Clone repository:**

    ```bash
    git clone <URL_repository_cua_ban>
    cd Askly
    ```

2.  **Tạo và kích hoạt môi trường ảo:**

    ```bash
    python -m venv venv
    # Trên Windows
    .\venv\Scripts\activate
    # Trên macOS/Linux
    source venv/bin/activate
    ```

3.  **Cài đặt các gói phụ thuộc:**

    Tạo một tệp `requirements.txt` với các thư viện cần thiết và chạy:

    ```bash
    pip install -r requirements.txt
    ```

    *Lưu ý: Các gói chính bao gồm `streamlit`, `gradio`, `tensorflow`, `tensorflow-hub`, `numpy`, `PyMuPDF`, `requests`.*

4.  **Cấu hình API Key (Tùy chọn):**

    Nếu bạn muốn sử dụng API Gemini của Google, hãy đặt API key của bạn trong tệp `config/config.py` hoặc dưới dạng biến môi trường.

    ```python
    # trong config/config.py
    API_KEY = "YOUR_GEMINI_API_KEY"
    ```

## Cách sử dụng

Dự án cung cấp hai giao diện. Bạn có thể chọn một trong hai để chạy.

### 1. Chạy với Streamlit

Giao diện Streamlit cung cấp một trải nghiệm giống như ứng dụng web truyền thống.

Để khởi chạy, hãy chạy lệnh sau từ thư mục gốc của dự án (`Askly`):

```bash
streamlit run app.py
```

### 2. Chạy với Gradio

Giao diện Gradio cung cấp một trải nghiệm giống chatbot hơn.

Để khởi chạy, hãy chạy lệnh sau từ thư mục gốc của dự án (`Askly`):

```bash
python gradio_app.py
```

Sau khi khởi chạy, hãy mở trình duyệt của bạn và truy cập vào URL cục bộ được cung cấp trong terminal (thường là `http://127.0.0.1:8765` cho Streamlit hoặc `http://127.0.0.1:7860` cho Gradio).

## Cấu trúc thư mục

```
Askly/
├── app.py                # Giao diện Streamlit
├── gradio_app.py         # Giao diện Gradio
├── rag_pipeline.py       # Luồng xử lý RAG chính
├── config/               # Các tệp cấu hình
├── data/                 # Lưu trữ PDF đã tải lên và embeddings
├── models/               # Quản lý embedding, LLM và hệ thống truy xuất
├── processors/           # Xử lý PDF và văn bản
└── utils/                # Các hàm tiện ích
```