# ✅ HOÀN THÀNH - Model Vi-Qwen2-1.5B-RAG Đã Sẵn Sàng!

## 🎉 Đã Tải Xong

Model **[Vi-Qwen2-1.5B-RAG](https://huggingface.co/AITeamVN/Vi-Qwen2-1.5B-RAG)** đã được tải về và cài đặt thành công!

```
✅ Model: Vi-Qwen2-1.5B-RAG (3GB)
✅ Location: models/model/
✅ Device: CUDA (GPU)
✅ Quantization: 4-bit (1.05GB VRAM)
✅ Status: READY TO USE!
```

---

## 🚀 Cách Chạy

### Start Backend + Frontend

```bash
conda activate rag311
bash start_all.sh
```

### Chỉ Backend

```bash
conda activate rag311
bash start_backend.sh
```

### Test CLI

```bash
conda activate rag311
python run.py
```

---

## 📊 Model Info

- **Name:** Vi-Qwen2-1.5B-RAG
- **Size:** 3GB (1.5B parameters)
- **Optimized for:** RAG tasks in Vietnamese
- **Context length:** Up to 8192 tokens
- **Source:** [AITeamVN](https://huggingface.co/AITeamVN/Vi-Qwen2-1.5B-RAG)

---

## 🔄 Thay Đổi Model Sau Này

Nếu muốn dùng model khác:

```bash
rm -rf models/model/*
cp -r /path/to/new/model/* models/model/
bash start_all.sh
```

---

**Đơn Giản Vậy Thôi!** 🎊
