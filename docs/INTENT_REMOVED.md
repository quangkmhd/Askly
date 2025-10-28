# Intent Classification Removed

## Thay đổi

Intent classification đã được loại bỏ hoàn toàn khỏi hệ thống RAG.

## Lý do

Hệ thống hiện sử dụng semantic search và keyword matching trực tiếp thay vì phân loại intent trước.

## Files đã thay đổi

1. **utils/query_processor.py** - Đã xóa `classify_intent()` và `intent_patterns`
2. **prompts/dynamic_prompts.py** - Đã xóa intent-based examples và instructions
3. **rag_pipeline.py** - Đã xóa bước classify intent
4. **models/retrieval_system.py** - Keyword search không còn dựa vào intent classification

## Tài liệu cũ

File `INTENT_ANALYSIS.md` chứa thông tin về intent classification cũ (đã không còn được sử dụng).

## Ngày thay đổi

28/10/2025
