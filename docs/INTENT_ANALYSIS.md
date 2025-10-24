# PHÂN TÍCH INTENT CHO HỆ THỐNG RAG - ĐẠI HỌC FPT

## 📊 HIỆN TRẠNG - 6 INTENTS CŨ

| Intent | Keywords | Use Case |
|--------|----------|----------|
| `tuition_fee` | học phí, chi phí, tiền học | Hỏi về học phí |
| `admission` | tuyển sinh, nhập học, xét tuyển | Hỏi về tuyển sinh |
| `grades` | điểm, gpa, điểm số | Hỏi về điểm, đánh giá |
| `schedule` | lịch, thời gian, học kỳ | Hỏi về lịch học |
| `graduation` | tốt nghiệp, điều kiện tốt nghiệp | Hỏi về tốt nghiệp |
| `program` | ngành, chuyên ngành, chương trình | Hỏi về ngành học |

---

## 📁 PHÂN TÍCH 16 PDFS TRONG HỆ THỐNG

### **Nhóm 1: NGHIÊN CỨU KHOA HỌC (4 files)**
1. **QĐ 1197** - Tiêu chuẩn năng lực và đạo đức trong NCKH, sở hữu trí tuệ
2. **QĐ 138** - Khen thưởng thành tích NCKH
3. **QĐ 338** - Quản lý đề tài NCKH cấp trường
4. **Chính sách hỗ trợ** - Tham dự hội nghị, hội thảo khoa học

**→ Đề xuất Intent: `research` (nghiên cứu khoa học)**

---

### **Nhóm 2: QUY ĐỊNH - NỘI QUY (4 files)**
5. **QD 1562** - Quy tắc ứng xử trường ĐH FPT
6. **QĐ 1234** - Nội quy ký thi
7. **QĐ 894** - Nội quy ký túc xá Hòa Lạc
8. **QĐ 736** - Quy chế đào tạo đại học chính quy

**→ Đề xuất Intents:**
- `conduct_rules` (quy tắc ứng xử, đạo đức)
- `exam_rules` (nội quy thi cử)
- `dormitory` (ký túc xá, chỗ ở)

---

### **Nhóm 3: ĐÀO TẠO & THẠC SĨ (2 files)**
9. **Quy chế đào tạo thạc sĩ** - Ban hành QC thạc sĩ
10. **Mô tả quá trình đào tạo** - Chi tiết quy trình

**→ Đề xuất Intent: `graduate_program` (thạc sĩ, sau đại học)**

---

### **Nhóm 4: THỰC TẬP & ĐỒ ÁN (1 file)**
11. **OJT + kì đồ án** - Quy định thực tập và đồ án

**→ Đề xuất Intent: `internship` (OJT, thực tập, đồ án)**

---

### **Nhóm 5: KHEN THƯỞNG (1 file)**
12. **QĐ 302** - Quy chế khen thưởng cuối học kỳ SV

**→ Đề xuất Intent: `awards` (khen thưởng, giải thưởng, học bổng)**

---

### **Nhóm 6: HỌC VỤ & THỦ TỤC (1 file)**
13. **Dữ liệu học vụ và thủ tục** - Thông tin hành chính

**→ Đề xuất Intent: `procedures` (thủ tục, hồ sơ, giấy tờ)**

---

### **Nhóm 7: CÔNG NGHỆ & HỆ THỐNG (1 file)**
14. **Dữ liệu công nghệ và hệ thống** - Hệ thống IT của trường

**→ Đề xuất Intent: `technology` (công nghệ, hệ thống, cơ sở vật chất)**

---

### **Nhóm 8: LIÊN HỆ (1 file)**
15. **Thông tin liên hệ** - Số điện thoại, email, địa chỉ

**→ Đề xuất Intent: `contact` (liên hệ, địa chỉ, hotline)**

---

## 🎯 TÓM TẮT: ĐỀ XUẤT 10 INTENTS MỚI

| # | Intent Name | Vietnamese | Keywords | Priority |
|---|-------------|------------|----------|----------|
| 1 | `research` | Nghiên cứu khoa học | nghiên cứu, nckh, đề tài, khoa học, hội nghị | 🔴 HIGH |
| 2 | `conduct_rules` | Quy tắc ứng xử | quy tắc, ứng xử, đạo đức, hành vi | 🟡 MEDIUM |
| 3 | `exam_rules` | Nội quy thi cử | thi, kiểm tra, nội quy thi, ký thi | 🔴 HIGH |
| 4 | `dormitory` | Ký túc xá | ký túc xá, ktx, chỗ ở, nội trú | 🔴 HIGH |
| 5 | `internship` | Thực tập & Đồ án | ojt, thực tập, đồ án, capstone | 🔴 HIGH |
| 6 | `awards` | Khen thưởng | khen thưởng, học bổng, giải thưởng | 🟡 MEDIUM |
| 7 | `procedures` | Thủ tục học vụ | thủ tục, hồ sơ, giấy tờ, hành chính | 🔴 HIGH |
| 8 | `contact` | Thông tin liên hệ | liên hệ, email, số điện thoại, địa chỉ | 🟢 LOW |
| 9 | `technology` | Công nghệ & HT | công nghệ, hệ thống, cơ sở vật chất | 🟢 LOW |
| 10 | `graduate_program` | Sau đại học | thạc sĩ, sau đại học, cao học | 🟡 MEDIUM |

---

## 💡 KEYWORDS CHO TỪNG INTENT

### **1. research** (Nghiên cứu khoa học)
```python
keywords = [
    # Tiếng Việt
    'nghiên cứu', 'nckh', 'đề tài', 'khoa học',
    'hội nghị', 'hội thảo', 'công bố', 'tạp chí',
    'sở hữu trí tuệ', 'bằng sáng chế', 'giải thưởng khoa học',
    'quản lý đề tài', 'thành tích nghiên cứu',
    # English
    'research', 'publication', 'conference', 'paper',
    'intellectual property', 'patent'
]
```

### **2. conduct_rules** (Quy tắc ứng xử)
```python
keywords = [
    'quy tắc', 'ứng xử', 'đạo đức', 'hành vi',
    'vi phạm', 'kỷ luật', 'xử lý', 'quy định ứng xử',
    'văn hóa', 'chuẩn mực',
    # English
    'conduct', 'behavior', 'ethics', 'violation'
]
```

### **3. exam_rules** (Nội quy thi cử)
```python
keywords = [
    'thi', 'kiểm tra', 'nội quy thi', 'ký thi',
    'gian lận', 'vi phạm thi', 'phòng thi',
    'giám thị', 'bài thi', 'đề thi',
    # English
    'exam', 'test', 'examination', 'cheating'
]
```

### **4. dormitory** (Ký túc xá)
```python
keywords = [
    'ký túc xá', 'ktx', 'chỗ ở', 'nội trú',
    'phòng ở', 'đăng ký ktx', 'hòa lạc',
    'nội quy ktx', 'tiện nghi', 'an ninh ktx',
    # English
    'dormitory', 'dorm', 'accommodation', 'housing'
]
```

### **5. internship** (Thực tập & Đồ án)
```python
keywords = [
    'ojt', 'thực tập', 'đồ án', 'capstone',
    'on-the-job training', 'dự án tốt nghiệp',
    'mentor', 'công ty thực tập', 'báo cáo thực tập',
    # English
    'internship', 'capstone project', 'final project'
]
```

### **6. awards** (Khen thưởng)
```python
keywords = [
    'khen thưởng', 'học bổng', 'giải thưởng',
    'sinh viên xuất sắc', 'sinh viên giỏi',
    'thành tích', 'danh hiệu', 'chứng chỉ',
    # English
    'award', 'scholarship', 'prize', 'honor'
]
```

### **7. procedures** (Thủ tục học vụ)
```python
keywords = [
    'thủ tục', 'hồ sơ', 'giấy tờ', 'hành chính',
    'đăng ký', 'xin', 'nộp', 'làm', 'cấp',
    'giấy xác nhận', 'bảng điểm', 'giấy chứng nhận',
    # English
    'procedure', 'document', 'paperwork', 'application'
]
```

### **8. contact** (Thông tin liên hệ)
```python
keywords = [
    'liên hệ', 'email', 'số điện thoại', 'địa chỉ',
    'hotline', 'phòng ban', 'văn phòng', 'tư vấn',
    'hỏi đáp', 'hỗ trợ', 'fanpage',
    # English
    'contact', 'phone', 'email', 'address', 'support'
]
```

### **9. technology** (Công nghệ & Hệ thống)
```python
keywords = [
    'công nghệ', 'hệ thống', 'cơ sở vật chất',
    'phòng lab', 'máy tính', 'wifi', 'mạng',
    'phần mềm', 'tài khoản', 'đăng nhập',
    # English
    'technology', 'system', 'infrastructure', 'lab', 'software'
]
```

### **10. graduate_program** (Sau đại học)
```python
keywords = [
    'thạc sĩ', 'sau đại học', 'cao học',
    'master', 'chương trình thạc sĩ',
    'đào tạo thạc sĩ', 'luận văn',
    # English
    'master', 'graduate', 'postgraduate', 'mba'
]
```

---

## 🔥 PRIORITY IMPLEMENTATION

### **Phase 1: HIGH PRIORITY (5 intents)**
Implement ngay - quan trọng nhất:
1. ✅ `research` - Nhiều tài liệu (4 files)
2. ✅ `exam_rules` - Sinh viên hỏi nhiều
3. ✅ `dormitory` - Use case cao
4. ✅ `internship` - Critical cho SV năm cuối
5. ✅ `procedures` - Câu hỏi phổ biến

### **Phase 2: MEDIUM PRIORITY (3 intents)**
Implement sau:
6. ⏳ `conduct_rules`
7. ⏳ `awards`
8. ⏳ `graduate_program`

### **Phase 3: LOW PRIORITY (2 intents)**
Implement cuối:
9. 🔵 `contact` - Simple, ít biến thể
10. 🔵 `technology` - Ít tài liệu

---

## 📝 SAMPLE QUERIES CHO TESTING

### **research**
- "Làm thế nào để đăng ký đề tài nghiên cứu khoa học?"
- "Quy định về sở hữu trí tuệ của trường ra sao?"
- "Có học bổng nào cho nghiên cứu khoa học không?"

### **exam_rules**
- "Nội quy phòng thi của trường như thế nào?"
- "Bị phát hiện gian lận thi cử thì xử lý ra sao?"
- "Cần chuẩn bị gì khi đi thi?"

### **dormitory**
- "Ký túc xá Hòa Lạc có điều kiện như thế nào?"
- "Làm sao để đăng ký ở KTX?"
- "Nội quy ký túc xá có những gì?"

### **internship**
- "OJT là gì?"
- "Làm đồ án capstone cần chuẩn bị gì?"
- "Khi nào sinh viên phải đi thực tập?"

### **procedures**
- "Làm giấy xác nhận sinh viên ở đâu?"
- "Thủ tục xin nghỉ học như thế nào?"
- "Cách xin bảng điểm ra sao?"

---

## 🎯 KẾT LUẬN

**Tổng cộng: 16 intents (6 cũ + 10 mới)**

Việc bổ sung 10 intents mới sẽ:
- ✅ Cover 100% nội dung 16 PDFs
- ✅ Xử lý chính xác hơn câu hỏi sinh viên
- ✅ Tăng độ liên quan của kết quả retrieval
- ✅ Hỗ trợ dynamic prompts tốt hơn

**Ưu tiên implement Phase 1 (5 intents HIGH) trước!**
