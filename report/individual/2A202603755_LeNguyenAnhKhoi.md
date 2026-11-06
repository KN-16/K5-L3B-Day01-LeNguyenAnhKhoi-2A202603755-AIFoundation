# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603755
- **Họ và tên:** Lê Nguyễn Anh Khôi
- **SEED:** `7979`

---

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc_1.jpg`](hand_calc_1.jpg) — ảnh tính tay có các bước của câu 1-3 | ☑ |
| [`hand_calc_2.jpg`](hand_calc_2.jpg) — ảnh tính tay có các bước của câu 4 | ☑ |
| [`hand_calc_3.jpg`](hand_calc_3.jpg) — ảnh tính tay có các bước của câu 5 | ☑ |

---

## 2. Kết quả chính

Bảng tổng hợp hiệu năng của 4 thế hệ AI trên tập dữ liệu phân tách theo `SEED = 7979` ($\text{train} = 3619$, $\text{val} = 775$, $\text{test} = 775$, $\text{SUB} = 40$):

| Phương pháp / Mô hình | F1 (test) | F1 (SUB 40) | p95 (ms/tin) | USD / 1k tin |
| --- | --- | --- | --- | --- |
| **1. BoW + LogReg (tự code)** | $0.888$ | $0.889$ | $0.109$ | $\$0.000$ |
| **2. BoW + MLP (deep learning)** | $0.952$ | $0.947$ | $0.352$ | $\$0.000$ |
| **3a. Encoder (MiniLM) + kNN** | $0.893$ | $0.976$ | $72.255$ | $\$0.000$ |
| **3b. Encoder (MiniLM) + LogReg** | $0.954$ | $1.000$ | $40.811$ | $\$0.000$ |
| **3c. Encoder (MiniLM) + MLP head** | $0.959$ | $1.000$ | $38.078$ | $\$0.000$ |
| **4a. LLM nhỏ zero-shot** (`gpt-oss-20b`) | $0.976$ | $0.976$ | $2448.439$ | $\$0.000^*$ |
| **4b. LLM lớn zero-shot** (`gpt-oss-120b`) | $1.000$ | $1.000$ | $2438.724$ | $\$0.000^*$ |
| **4c. LLM nhỏ few-shot** (`gpt-oss-20b`) | $0.947$ | $0.947$ | $2505.423$ | $\$0.000^*$ |

*Phần 4 sử dụng nhà cung cấp `groq` (gói miễn phí nên chi phí ghi nhận trong dict `PRICE_PER_1M` là `$0.0`; nếu quy chiếu theo đơn giá chuẩn của model tương đương `gpt-6-luna` `($0.10/$0.50)` mỗi `1M` token thì bản zero-shot tốn ≈ \$0.006 - \$0.007/1k tin và bản few-shot tốn ≈ \$0.022/1k tin).*

---

## 3. Dự đoán và thực tế

Ba dự đoán `predict(...)` có sự khác biệt đáng chú ý nhất so với kết quả chạy thực tế:

1. **Hiệu năng cơ sở của Bag-of-Words (`ml_f1`):**
   - **Dự đoán:** $\text{F1} \approx 0.70$, trong đó $\text{Precision} > \text{Recall}$ do dữ liệu lệch nhiều về lớp `ham` khiến mô hình dễ đoán thiếu `spam` ($\text{FN}$ cao).
   - **Thực tế:** Chiều hướng $\text{Precision} > \text{Recall}$ hoàn toàn đúng ($\text{Precision} = 1.000$, $\text{Recall} = 0.798$), nhưng điểm $\text{F1}$ trên tập test đạt tới **$0.888$** (cao hơn rất nhiều so với mức $0.70$).
   - **Lý do lệch:** Tập dữ liệu SMS gốc của UCI chứa nhiều tin rác theo khuôn mẫu lặp lại với các từ khóa quảng cáo rất lộ liễu (`call`, `txt`, `free`, `claim`, `150p`). Chỉ cần bộ từ điển nhị phân $2000$ từ và mô hình tuyến tính đơn giản đã đủ để tách biệt phần lớn các mẫu chuẩn này.

2. **Hành vi của BoW trước tin nhắn ngụy trang và bẫy từ khóa (`emb_tricky` & `hard_eval`):**
   - **Dự đoán:** Ở `emb_tricky`, dự đoán `3b. Encoder + LogReg` sẽ thắng rõ `1. BoW + LogReg`. Ở `hard_eval`, dự đoán BoW sẽ sập nặng ở nhóm tin thường chứa từ khóa nhạy cảm (`ham_looks_spam` và `talks_about_spam`).
   - **Thực tế:** Ở `emb_tricky`, cả BoW và `Encoder + LogReg` đều hòa nhau với **$5/6$ câu đúng** (cùng sai câu số 6). Ở `hard_eval`, BoW lại đoán đúng **$100\%$** ở cả hai nhóm `ham_looks_spam` và `talks_about_spam`, nhưng lại thất bại gần như hoàn toàn ở nhóm `obfuscation` ($0\% - 17\%$) và `natural_spam` ($10\% - 20\%$).
   - **Lý do lệch:** Do dữ liệu huấn luyện có $\sim 87.4\%$ là `ham`, hệ số chệch $b$ và các đại từ giao tiếp đời thường (`i`, `you`, `me`, `my`) mang trọng số âm rất mạnh kéo dự đoán về `ham`. Nhờ "thiên kiến đoán ham" này mà các tin nhắn bạn bè dù chứa 1–2 từ khóa `free`, `won` vẫn không vượt qua được ngưỡng $0.5$ để bị bắt nhầm; nhưng đổi lại, khi gặp tin rác viết giọng tự nhiên hoặc biến tấu ký tự làm mất từ khóa, vector đặc trưng rơi về gần $\vec{0}$ khiến mô hình mặc định đoán là `ham` và bỏ lọt toàn bộ.

3. **Hiệu quả của kỹ thuật Few-shot trên LLM nhỏ (`llm_fewshot`):**
   - **Dự đoán:** Việc cung cấp thêm 4 ví dụ mẫu (2 spam, 2 ham) vào prompt sẽ giúp $\text{F1}$ tăng nhẹ (từ $+0.01$ đến $+0.03$) hoặc giữ nguyên mức tối đa so với zero-shot.
   - **Thực tế:** Điểm $\text{F1}$ trên tập `SUB 40` của `4c. LLM nhỏ few-shot` lại **giảm nhẹ từ $0.976$ xuống $0.947$** ($\text{Precision}$ tăng lên $1.000$ nhưng $\text{Recall}$ giảm từ $1.000$ xuống $0.900$), đồng thời độ trễ trung vị $p50$ tăng vọt gấp $6.6\text{ lần}$ (từ $344.28\text{ ms}$ lên $2293.18\text{ ms}$).
   - **Lý do lệch:** Hai ví dụ spam lấy tự động ở đầu tập train đều thuộc dạng "câu đố trúng thưởng" (`Sunshine Quiz`, `Moby Pub Quiz`), vô tình tạo thiên kiến ngữ cảnh khiến model nhỏ (`gpt-oss-20b`) thu hẹp tiêu chí nhận diện spam và bỏ sót các mẫu spam kiểu khác trong tập SUB. Ngoài ra, prompt dài gấp 3–4 lần làm tăng thời gian xử lý và suy luận của model.

---

## 4. Phân tích 50 case khó

### Mô hình sụt giảm mạnh nhất
Khi chuyển từ tập test tiêu chuẩn (cùng phân phối với tập train) sang bộ 50 case khó viết tay, **nhóm mô hình dựa trên Bag-of-Words tụt điểm nghiêm trọng nhất**:
- **`2. BoW + MLP`:** $\text{F1}$ tụt **$0.526$ điểm** (từ $0.947$ xuống $0.421$, Accuracy chỉ đạt $56\%$).
- **`1. BoW + LogReg`:** $\text{F1}$ tụt **$0.511$ điểm** (từ $0.889$ xuống $0.378$, Accuracy đạt $54\%$).
- **Nhóm Embedding (`3a, 3b, 3c`):** $\text{F1}$ giảm ở mức trung bình (từ $0.976 - 1.000$ xuống còn $0.625 - 0.746$, Accuracy đạt $64\% - 70\%$). Đáng chú ý, `3c. Encoder + MLP head` ($64\%$) tụt sâu hơn `3b. Encoder + LogReg` ($68\%$) và `3a. kNN` ($70\%$) do tầng ẩn MLP bị quá khớp (overfit) vào ranh giới của tập train chuẩn.
- **Nhóm LLM (`4a, 4d`):** Bền vững nhất, giữ được $\text{F1}$ từ $0.918 - 0.933$ (Accuracy đạt $90\% - 92\%$). Prompt phòng thủ (`4d`) giúp chặn đứng hoàn toàn nhóm **Prompt injection** (tăng từ $83\%$ ở `4a` lên **$100\%$** ở `4d`).

### Phân tích 2 case điển hình và cơ chế gây lỗi

1. **Case #30 — Nhóm Né từ khoá (`obfuscation`, nhãn thật = `spam`):**
   > *"F R E E  e n t r y  into our weekly draw, t x t  WIN  to 80086"*
   - **Mô hình sai:** Cả `1. BoW + LogReg` và `2. BoW + MLP` đều đoán sai thành `ham` (nhóm này MLP đúng $0\%$, LogReg đúng $17\%$).
   - **Cơ chế gây lỗi:** Hàm `tokenize` tách từ bằng biểu thức chính quy `r"[a-z0-9£$€]+"`. Khi kẻ phát tán spam chèn khoảng trắng vào giữa từng chữ cái (`F R E E`, `t x t`) hoặc thay chữ bằng số/ký hiệu ở các câu cùng nhóm (`C0ngr4tulat10ns`, `c.a.s.h`), chuỗi bị băm nhỏ thành các token ký tự đơn lẻ (`"f"`, `"r"`, `"e"`, `"t"`, `"x"`) hoặc từ lạ không nằm trong từ điển `vocab` ($2000$ từ). Hệ quả là các chiều trọng số dương lớn của từ khóa spam không được kích hoạt, làm tổng $\mathbf{w} \cdot \mathbf{x} + b < 0$ và mô hình phán đoán nhầm là `ham`.

2. **Case #47 — Nhóm Nói về spam (`talks_about_spam`, nhãn thật = `ham`):**
   > *"lol I just got a text saying I won £1000 and to call some 0906 number, people still fall for that?"*
   - **Mô hình sai:** Cả 3 mô hình Embedding (`3a, 3b, 3c`) và mô hình `4d. LLM nhỏ + prompt phòng thủ` đều đoán sai thành `spam`.
   - **Cơ chế gây lỗi:** Mô hình `MiniLM` nén toàn bộ câu thành một vector 384 chiều phản ánh chủ đề ngữ nghĩa tổng quát. Do câu trích dẫn nguyên văn cụm `"won £1000"` và `"call some 0906 number"`, vector nhúng của câu bị kéo mạnh về vùng không gian của các tin nhắn trúng thưởng thật; bộ Encoder kích thước nhỏ không phân biệt được mệnh đề bao ngoài mang tính chất *châm biếm/kể lại* (`"lol I just got a text..."`, `"people still fall for that?"`). Tương tự, với `4d`, việc đặt chỉ dẫn phòng thủ nghiêm ngặt khiến LLM nhỏ trở nên quá nhạy cảm với các dấu hiệu giải thưởng/số điện thoại và đánh nhãn nhầm là `spam`.

---

## 5. Thử thách cá nhân

- **Đề bài (theo `SEED = 7979` $\rightarrow$ Đề #6):** *LLM · Sửa `build_classification_prompt` để LLM trả JSON `{"label": ..., "reason": ...}`, viết parse cho JSON, chạy lại trên 10 tin SUB. So token/chi phí với bản 1 từ. Đáng không?*
- **Cách thực hiện:**
  1. Xây dựng hàm `build_json_prompt(text, examples)` yêu cầu LLM chỉ trả về duy nhất một đối tượng JSON hợp lệ theo cấu trúc `{"label": "spam" or "ham", "reason": "brief explanation in 1 short sentence"}`.
  2. Xây dựng hàm `parse_json_label(response)` sử dụng regex `r"\{.*\}"` (`re.DOTALL`) để bóc tách chuỗi JSON, dùng `json.loads()` đọc giá trị khóa `"label"` quy đổi về `1` (`spam`) hoặc `0` (`ham`), đồng thời bọc `try...except` gọi lại `parse_label()` làm phương án dự phòng nếu JSON bị lỗi cú pháp.
  3. Nâng tạm thời `CLASSIFY_MAX_TOKENS` lên $150$ token để JSON không bị cắt cụt giữa chừng, sau đó chạy thực nghiệm trên 10 mẫu đầu tiên của tập `SUB` với model `openai/gpt-oss-20b`.
- **Kết quả thực nghiệm (trên 10 tin SUB):**
  - **Bản trả lời 1 từ:** $\text{F1} = 1.000$ | Tổng độ trễ $= 24.05\text{ s}$ | Token ước tính (In / Out) $\approx 545 / 13$ | Chi phí quy đổi $\approx \$0.000061$.
  - **Bản trả lời JSON:** $\text{F1} = 1.000$ | Tổng độ trễ $= 23.51\text{ s}$ | Token ước tính (In / Out) $\approx 679 / 161$ | Chi phí quy đổi $\approx \$0.000149$.
  - **Tỷ lệ chênh lệch:** Lượng Output token tăng gấp **$12.1\text{ lần}$**, làm tổng chi phí tăng gấp **$2.4\text{ lần}$**.
- **Đánh giá & Giải thích:**
  - Việc bắt LLM trả về JSON kèm lý do **không làm tăng độ chính xác phân loại** ($\text{F1}$ giữ nguyên $1.000$), nhưng lại làm phình lượng token đầu ra gấp hơn 12 lần. Vì các nhà cung cấp API định giá output token đắt gấp $4 - 5\text{ lần}$ input token, chi phí vận hành bị đội lên gấp $2.4 - 3.5\text{ lần}$, kèm theo nguy cơ lỗi định dạng JSON nếu câu giải thích dài quá `max_tokens`.
  - Do đó, phương án này **không đáng** dùng cho luồng lọc tự động quy mô lớn trên production (nơi chỉ cần nhãn `0/1` với chi phí tối thiểu), mà **chỉ đáng áp dụng** cho công cụ kiểm toán nội bộ (auditing/debugging) hoặc tính năng hiển thị lời giải thích ngắn gọn giúp người dùng cuối hiểu vì sao tin nhắn bị cảnh báo.

---

## 6. Phương án cho sản phẩm thật

### Bài toán vận hành
Hệ thống nhà mạng cần xử lý **$10\text{ triệu SMS/ngày}$** (trung bình $\approx 116\text{ tin/giây}$, giờ cao điểm có thể đạt $300 - 500\text{ tin/giây}$) với cam kết SLA độ trễ **$p95 < 50\text{ ms/tin}$**.

### Đối chiếu số liệu và Lựa chọn kiến trúc
Dựa trên bảng kết quả thực nghiệm:
- Các mô hình đạt chuẩn SLA ($p95 < 50\text{ ms}$): **`1. BoW + LogReg`** ($0.109\text{ ms}$), **`2. BoW + MLP`** ($0.352\text{ ms}$), **`3b. Encoder + LogReg`** ($40.811\text{ ms}$) và **`3c. Encoder + MLP head`** ($38.078\text{ ms}$). Phương án `3a. kNN` ($72.255\text{ ms}$) và toàn bộ các phương án gọi **`4. LLM`** ($p95 \approx 2438 - 2505\text{ ms}$, chi phí ước tính $\approx \$2,100 - \$21,000\text{/tháng}$ cho $300\text{ triệu tin}$) đều vi phạm nặng nề SLA độ trễ.
- Tuy nhiên, nếu chỉ dùng `2. BoW + MLP` thì dễ bị qua mặt bởi spam biến tướng (F1 trên 50 case khó chỉ đạt $0.421$), còn nếu đẩy toàn bộ $116\text{ tin/s}$ qua `3b. Encoder + LogReg` thì thông lượng đơn luồng ($31.04\text{ tin/s}$) không đủ đáp ứng ngay trên 1 worker và độ trễ sát ngưỡng trần SLA ($40.81\text{ ms}$).

$\Rightarrow$ **Giải pháp lựa chọn: Hệ thống phân tầng lai (Cascading Router 3 tầng)** kết hợp tiền xử lý chuẩn hóa văn bản:

1. **Bước 0 — Tiền xử lý chuẩn hóa (De-obfuscation Regex):** Ghép các ký tự rời rạc (`F R E E`, `c.a.s.h` $\rightarrow$ `free`, `cash`) và quy đổi ký tự đồng형 (`0` $\rightarrow$ `o`, `1` $\rightarrow$ `i`, `4` $\rightarrow$ `a`, `€` $\rightarrow$ `e`) để khắc phục điểm mù né từ khóa trong $< 0.05\text{ ms}$.
2. **Tầng 1 — Phễu lọc tốc độ cao bằng `2. BoW + MLP` ($p95 = 0.352\text{ ms}$, $28,098\text{ tin/s}$):**
   - Tiếp nhận $100\%$ lưu lượng ($116\text{ tin/s}$). Với các tin nhắn có xác suất rõ ràng ($p(\text{spam}) < 0.15 \rightarrow \text{ham}$, $p(\text{spam}) > 0.85 \rightarrow \text{spam}$), hệ thống trả kết quả ngay lập tức. Tầng này giải quyết gọn $\sim 85\% - 90\%$ lưu lượng giao tiếp thông thường và spam đại trà với chi phí gần như bằng $0$.
3. **Tầng 2 — Phân loại ngữ nghĩa bằng `3b. Encoder (MiniLM) + LogReg` ($p95 = 40.811\text{ ms}$, $31.04\text{ tin/s}$):**
   - Chỉ xử lý $\sim 10\% - 15\%$ lượng tin nhắn "lưỡng lự" ($p \in [0.15, 0.85]$) do Tầng 1 chuyển xuống (ưu tiên chọn `3b` thay vì `3c` vì đầu phân loại tuyến tính bền vững hơn trên tập Stress Test: $68\%$ so với $64\%$).
   - Khi đó, lưu lượng thực tế đi vào mạng Encoder chỉ còn $\sim 12 - 17\text{ tin/s}$ (nằm gọn trong năng lực $31\text{ tin/s}$ của 1 tiến trình), và tổng độ trễ $p95$ cộng dồn của cả 2 tầng là $0.352 + 40.811 = \mathbf{41.163\text{ ms} < 50\text{ ms}}$ (đảm bảo tuân thủ SLA).
4. **Tầng 3 — Kiểm toán bất đồng bộ bằng `4d. LLM nhỏ + Prompt phòng thủ` (Offline Active Learning):**
   - Không nằm trên luồng chặn thời gian thực. Chỉ lấy mẫu $< 0.5\%$ các tin nhắn cực kỳ nhập nhằng ở Tầng 2 ($p \in [0.45, 0.55]$) đẩy vào hàng đợi nền để LLM phân tích sâu, phát hiện mẫu lừa đảo mới và tự động gán nhãn bổ sung dữ liệu huấn luyện định kỳ cho Tầng 1 & 2.

- **Đánh đổi chấp nhận:** Chấp nhận kiến trúc hệ thống phức tạp hơn (quản lý ngưỡng chuyển tầng và pipeline bất đồng bộ) cũng như độ trễ $\sim 41\text{ ms}$ ở phân khúc $15\%$ tin nhắn khó, để đổi lấy khả năng đáp ứng chặt chẽ SLA $< 50\text{ ms}$, tiết kiệm hơn $95\%$ chi phí hạ tầng/API và chống chịu tốt trước các kỹ thuật lách từ khóa hay prompt injection.

---

## 7. AI log và điều đã học

### Nhật ký sử dụng AI (AI Log)

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| **Phần 4 (TODO 4.1 – 4.4)** | Cung cấp chữ ký hàm và docstring của 4 hàm (`call_openai`, `build_classification_prompt`, `parse_label`, `llm_classify_batch`), yêu cầu viết code Python gọi Chat Completions API, tạo prompt phân loại zero/few-shot, lọc nhãn trả về và chạy phân loại theo lô kèm tính chi phí token. | AI sinh code khung cho cả 4 hàm: khởi tạo `OpenAI()` và đo thời gian bằng `perf_counter`, ghép chuỗi prompt với các ví dụ mẫu, dùng biểu thức chính quy (`re`) để tìm nhãn và vòng lặp tính tổng thời gian, chi phí. | Tự rà soát và tinh chỉnh: (1) Đảm bảo truyền `**provider_kwargs(max_tokens)` và thêm `or ""` ở `call_openai` để tương thích với reasoning model (Groq/OpenAI) và tránh lỗi khi content trả về `None`; (2) Ở `parse_label`, kiểm tra cụm `"not spam"` trước khi dùng regex `\bspam\b` để tránh bắt nhầm nhãn phủ định; (3) Kiểm tra việc truyền tham số `prompt_fn` và `retry_with_backoff` trong `llm_classify_batch`, sau đó xác nhận bằng unit test giả lập (`unittest.mock`) và chạy thật với Groq API. |

### Điều đã học được hôm nay
Qua việc tự tay cài đặt cả 4 thế hệ AI trên cùng một bài toán, tôi nhận ra rằng điểm số $\text{F1}$ cao ($0.95+$) trên tập test tiêu chuẩn rất dễ gây ảo tưởng vì dữ liệu kiểm thử cùng phân phối với dữ liệu huấn luyện; khi đưa vào môi trường đối kháng (50 case khó), mô hình phức tạp hơn (như `MLP head` nhiều tham số hay `few-shot` chọn mẫu lệch) thậm chí còn kém bền vững hơn mô hình tuyến tính đơn giản (`LogReg`). Đồng thời, đối với các dòng *reasoning LLM* hiện đại, việc kiểm soát tham số suy luận (`reasoning_effort`, hạn mức `max_tokens`) và tách biệt ranh giới dữ liệu bằng thẻ `<sms>` là bắt buộc để tránh cạn kiệt token ngầm hoặc bị tấn công *prompt injection*. Quan trọng nhất, không có mô hình đơn lẻ nào tối ưu toàn diện cả về độ chính xác ngữ nghĩa, độ trễ $p95$ lẫn chi phí, nên tư duy thiết kế hệ thống phân tầng (Cascading Router) mới là chìa khóa đưa AI vào sản phẩm thực tế.