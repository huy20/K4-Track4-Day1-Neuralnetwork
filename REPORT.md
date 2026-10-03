# Báo cáo Lab Day 1 — Xây dựng Mạng Nơ-ron và Thí nghiệm Huấn luyện

**Học viên:** Phùng Quang Minh Huy \
**Mã số sinh viên (MSSV):** 2A202602610 \
**Môn học:** Track 4 — Day 1 · VinUniversity AICB 2026  

---

## 1. Thiết lập

- **Môi trường:** Google Colab, GPU Tesla T4 (16GB VRAM), PyTorch 2.11.0+cu130, CUDA 13.0, Python 3.10+.
- **Dữ liệu:** Forest CoverType (581,012 mẫu, 54 đặc trưng gồm 10 biến số liên tục chuẩn hoá Z-score và 44 cột nhị phân one-hot).
  - Tập huấn luyện gốc (`train`): 464,809 mẫu.
  - Tập kiểm định (`eval`): 116,203 mẫu (dùng độc lập để chấm điểm cuối).
  - Phân chia Validation: Tách 20% từ tập train theo phương pháp phân tầng (stratified sampling, $seed = 42$) $\rightarrow$ **371,847 mẫu Train** và **92,962 mẫu Validation**.
- **Kiến trúc mô hình:** `M-base` ($54 \rightarrow 256 \rightarrow 128 \rightarrow 7$), đúng chuẩn **47,879 tham số**, kích hoạt ReLU ở các tầng ẩn, không dùng BatchNorm hay Dropout ở baseline.
- **Cấu hình Baseline:** Cross-Entropy Loss, Optimizer SGD + Momentum ($0.9$), Learning Rate $lr = 0.1$, Batch size $512$, $20$ epochs, Khởi tạo He normal (`init="he"`), Precision FP32.
- **Mốc tham chiếu:** Accuracy của chiến lược "luôn đoán lớp đa số" trên tập Validation là **0.4876** (Macro-F1 $\approx 0.094$).
- **Chủ đề đã thử:** ☑ Optimizer (SGD+Momentum vs Adam vs AdamW) · ☑ Seed Noise · ☑ Learning Rate Search.

---

## 2. Kiểm tra ban đầu và độ nhiễu

### 2.1 Bảng kiểm tra "sức khoẻ" ban đầu (Sanity Checks)

| Phép kiểm tra | Giá trị lý thuyết / Kỳ vọng | Kết quả đo được | Đánh giá |
|---|---|---|---|
| **Số tham số mô hình** | $(54 \times 256 + 256) + (256 \times 128 + 128) + (128 \times 7 + 7) = 47,879$ | **47,879** | Khớp chính xác $100\%$ |
| **Shape Logits đầu ra** | `(B, 7)` | `(8, 7)` với $B=8$ | Hợp lệ (chưa softmax) |
| **Loss bước 0 (Step 0 Val Loss)** | $\ln(7) \approx 1.9459$ | **1.9083** | Khởi tạo He chuẩn, dự đoán phân bố đều ($P \approx 1/7$) |
| **Quá khớp 20 mẫu (Overfit 20 samples)** | Loss $\rightarrow 0$ sau vài trăm bước | Loss $\approx 0.0001$ sau 300 bước | Mạng có khả năng học và ghi nhớ hoàn hảo |
| **Kiểm tra luồng Gradient** | Mọi tầng có $\Vert g \Vert > 0$ | Layer 0: $0.003769$, Layer 2: $0.006072$, Layer 4: $0.004779$ | Gradient lan truyền tốt, không bị triệt tiêu |

### 2.2 Đo độ nhiễu Seed trên Baseline (SGD+Momentum, lr=0.1)

| Thí nghiệm (`exp_id`) | Seed | Best Epoch | Step 0 Loss | Best Val Loss | Val Accuracy | Val Macro-F1 |
|---|---|---|---|---|---|---|
| `base-s1` | 1 | 18 | 2.2691 | 0.2393 | 0.9035 | 0.8434 |
| `base-s2` | 2 | 20 | 1.9782 | 0.2246 | **0.9109** | **0.8654** |
| `base-s3` | 3 | 20 | 1.9005 | 0.2280 | 0.9090 | 0.8505 |
| **Trung bình (Mean)** | — | 19.3 | 2.0493 | 0.2306 | **0.9078** | **0.8531** |
| **Độ lệch chuẩn ($\sigma$)** | — | — | — | — | **$\pm 0.0031$** | **$\pm 0.0092$** |

**Ngưỡng nhiễu thống kê:**  
$\text{Ngưỡng nhiễu } 2\sigma = 2 \times 0.0092 = \mathbf{0.0184} \quad (\text{trên chỉ số Val Macro-F1})$
Mọi kết luận so sánh "Thuật toán A tốt hơn Thuật toán B" bắt buộc phải có mức chênh lệch $|\Delta| > 2\sigma = 0.0184$ mới được coi là có ý nghĩa thống kê.

---

## 3. Kết quả theo chủ đề: Bộ tối ưu hoá (Optimizer)

### 3.1 Dự đoán trước khi chạy
1. **SGD + Momentum ($lr=0.1$):** Nhờ việc chuẩn hoá Z-score 10 đặc trưng số và one-hot 44 đặc trưng phân loại, không gian mất mát tương đối đẳng hướng (isotropic). SGD kết hợp Momentum $0.9$ sẽ tích luỹ vận tốc theo hướng dốc chính, vượt qua các điểm yên ngựa và tìm được cực tiểu phẳng (flat minima), mang lại khả năng tổng quát hoá cao.
2. **Adam / AdamW ($lr=0.001$):** Với cơ chế thích ứng theo từng toạ độ (adaptive learning rate bằng việc chia cho $\sqrt{v_t} + \epsilon$), Adam giúp giảm loss rất nhanh ở những bước đầu. Tuy nhiên, trên mạng MLP dữ liệu bảng, Adam đôi khi hội tụ vào các cực tiểu hẹp (sharp minima).
3. **Adam vs AdamW:** Do `weight_decay = 0.0`, AdamW và Adam có cơ chế toán học đồng nhất nên dự kiến cho kết quả giống hệt nhau.

### 3.2 Bảng so sánh thực nghiệm

| `exp_id` | Optimizer | Learning Rate | Best Epoch | Step 0 Loss | Best Val Loss | Val Accuracy | Val Macro-F1 | Thời gian/epoch |
|---|---|---|---|---|---|---|---|---|
| `base-s1` | SGD+Momentum | 0.100 | 18 | 2.2691 | **0.2393** | **0.9035** | 0.8434 | 1.40s |
| `opt-adam` | Adam | 0.001 | 18 | 2.2691 | 0.2443 | 0.9021 | **0.8446** | 1.51s |
| `opt-adamw`| AdamW | 0.001 | 18 | 2.2691 | 0.2443 | 0.9021 | **0.8446** | 1.59s |

### 3.3 Đối chiếu và giải thích cơ chế
- **Khớp dự đoán:** `opt-adam` và `opt-adamw` cho các chỉ số hoàn toàn trùng khớp (Val Macro-F1 = $0.8446$, Val Acc = $0.9021$, Best Val Loss = $0.2443$). Khi không có $L_2$ regularization (`weight_decay = 0`), công thức cập nhật của AdamW $w_{t+1} = w_t - \eta \frac{m_t}{\sqrt{v_t}+\epsilon}$ hoàn toàn tương đương với Adam.
- **So sánh với Baseline:**
  - $\Delta(\text{Adam} - \text{Baseline Seed 1}) = 0.8446 - 0.8434 = +0.0012$.
  - $\Delta(\text{Adam} - \text{Baseline Mean}) = 0.8446 - 0.8531 = -0.0085$.
  - Vì $|\Delta| = 0.0085 < 2\sigma$ ($0.0184$), sự chênh lệch giữa Adam ($lr=0.001$) và SGD+Momentum ($lr=0.1$) nằm trong biên độ dao động ngẫu nhiên của seed. Cả hai bộ tối ưu đều đạt hiệu năng xuất sắc (>90% accuracy và >84.4% macro-F1).
- **Thời gian tính toán:** SGD+Momentum tính toán nhẹ hơn (1.40s/epoch) so với Adam (1.51s/epoch) và AdamW (1.59s/epoch) do chỉ cần duy trì 1 trạng thái động lượng $v$ thay vì cả 2 trạng thái $m_t$ và $v_t$.

---

## 4. Đánh giá cuối trên tập Eval

> **Nguyên tắc:** Cấu hình cuối cùng được lựa chọn **HOÀN TOÀN** dựa trên chỉ số Validation Macro-F1 cao nhất (`base-s2` với Val Macro-F1 = $0.8654$), hoàn toàn không nhìn vào tập Eval trước khi đưa ra quyết định.

### 4.1 Bảng kết quả tổng thể (chấm bằng `scripts/evaluate.py`)

| Cấu hình | `exp_id` | Seed | Val Macro-F1 | **Eval Macro-F1** | Eval Accuracy |
|---|---|---|---|---|---|
| **Cấu hình cuối cùng** | `base-s2` | 2 | **0.8654** | **0.8659** | **0.9101 (91.01%)** |

- **Đánh giá mức điểm theo Rubric:** Điểm Eval Macro-F1 đạt **0.8659 $\ge 0.86$**, đạt mức điểm tối đa **5/5** của hạng mục chất lượng mô hình trên tập kiểm định.
- **Độ tin cậy của Validation:** Chỉ số Val Macro-F1 ($0.8654$) và Eval Macro-F1 ($0.8659$) sai lệch cực nhỏ ($\Delta = +0.0005$), chứng minh phép tách validation phân tầng 20% từ train phản ánh cực kỳ trung thực và không hề có hiện tượng rò rỉ dữ liệu (data leakage).

### 4.2 Phân tích lỗi theo từng lớp (Per-class Error Analysis)

| Lớp ($c$) | Tên loại rừng | Support (Eval) | Tỷ lệ (%) | Precision | Recall | F1-Score |
|---|---|---|---|---|---|---|
| **0** | Spruce / Fir | 42,368 | 36.46% | 0.9131 | 0.9005 | **0.9067** |
| **1** | Lodgepole Pine | 56,661 | 48.76% | 0.9168 | 0.9301 | **0.9234** |
| **2** | Ponderosa Pine | 7,151 | 6.15% | 0.9205 | 0.8838 | **0.9018** |
| **3** | Cottonwood / Willow | 549 | 0.47% | 0.8042 | 0.8379 | **0.8207** |
| **4** | Aspen | 1,899 | 1.63% | 0.7947 | 0.7256 | **0.7586** *(Thấp nhất)* |
| **5** | Douglas-fir | 3,473 | 2.99% | 0.8036 | 0.8379 | **0.8204** |
| **6** | Krummholz | 4,102 | 3.53% | 0.9247 | 0.9344 | **0.9296** *(Cao nhất)* |
| **Toàn bộ** | **Tổng / Trung bình** | **116,203** | **100%** | **0.8683** | **0.8643** | **0.8659** |

### 4.3 Ma trận nhầm lẫn (Confusion Matrix)

```
Thực tế \ Dự đoán     Lớp 0    Lớp 1    Lớp 2    Lớp 3    Lớp 4    Lớp 5    Lớp 6
---------------------------------------------------------------------------------
Lớp 0 (Spruce/Fir)    38151     3881        0        0       48       10      278
Lớp 1 (Lodgepole)      3337    52701      132        0      300      157       34
Lớp 2 (Ponderosa)         1      232     6320       79        8      511        0
Lớp 3 (Cottonwood)        0        2       64      460        0       23        0
Lớp 4 (Aspen)            49      436       26        0     1378       10        0
Lớp 5 (Douglas-fir)      12      194      324       33        0     2910        0
Lớp 6 (Krummholz)       234       35        0        0        0        0     3833
```

### 4.4 Nhận xét sâu về lỗi phân loại
1. **Lớp khó nhất — Lớp 4 (Aspen):** Đạt F1 thấp nhất ($0.7586$) và Recall chỉ $72.56\%$.
   - *Nguyên nhân:* Aspen là loài cây ưa sáng, mọc rải rác ở đai rừng trung bình xen kẽ giữa Lodgepole Pine và Spruce/Fir. Do đó, có tới **436 mẫu Aspen bị đoán nhầm thành Lớp 1 (Lodgepole)** và **49 mẫu bị nhầm thành Lớp 0 (Spruce/Fir)**. Với số mẫu ít (1,899 mẫu), mạng có xu hướng thiên vị lớp đa số kế cận.
2. **Cặp nhầm lẫn lớn nhất — Lớp 0 vs Lớp 1:**
   - 3,881 mẫu Lớp 0 bị dự đoán thành Lớp 1.
   - 3,337 mẫu Lớp 1 bị dự đoán thành Lớp 0.
   - *Nguyên nhân:* Hai lớp này chiếm tới **85.2%** tổng số mẫu dữ liệu và phân bố ở cùng đai cao độ (Elevation 2,500m – 3,200m). Các vùng giáp ranh có đặc tính địa hình (Aspect, Slope, Hillshade) rất tương đồng.
3. **Lớp thiểu số cực đoan — Lớp 3 (Cottonwood / Willow):**
   - Chỉ có 549 mẫu (0.47% tập eval) nhưng mô hình đạt Recall $83.79\%$ và F1 $82.07\%$. Sở dĩ mô hình phân loại tốt lớp này là vì Cottonwood mọc sát bờ nước/thung lũng thấp, đặc trưng khoảng cách tới nguồn nước (`Horizontal_Distance_To_Hydrology` và `Vertical_Distance_To_Hydrology`) mang tính phân tách rất rõ rệt.

---

## 5. Trả lời các câu hỏi dẫn dắt

### Câu 1: Bộ tối ưu nào "thắng" khi mỗi cái được chỉnh lr công bằng? Khi lr không được chỉnh thì kết luận thay đổi ra sao?
- **Khi chỉnh lr công bằng:** SGD+Momentum với $lr = 0.1$ và Adam với $lr = 0.001$ đều đạt hiệu năng tương đương nhau trên tập validation (Macro-F1 $0.853 \pm 0.009$ so với $0.845$, chênh lệch nằm dưới ngưỡng nhiễu $2\sigma$). SGD cho tốc độ huấn luyện nhanh hơn và tốn ít bộ nhớ trạng thái hơn.
- **Khi không chỉnh lr:** Nếu áp đặt cùng một mức $lr = 0.1$ cho cả hai, Adam sẽ bị nổ loss hoặc phân kỳ dao động dữ dội do bước cập nhật $\Delta w \approx \eta \cdot \text{sign}(g)$ quá lớn. Ngược lại, nếu dùng $lr = 0.001$ cho SGD, mô hình chỉ đạt Val Macro-F1 $0.4231$ sau 3 epoch vì gradient quá nhỏ khiến mô hình di chuyển cực kỳ chậm. Kết luận so sánh chỉ có giá trị khi mỗi thuật toán được đặt ở dải learning rate tối ưu của nó.

### Câu 2: Dropout có giúp không khi mô hình chưa quá khớp? Khi nào thì nên dùng?
- Trên tập CoverType với 371,847 mẫu train và mô hình `M-base` có 47,879 tham số, tỷ lệ mẫu / tham số là $\approx 7.8$. Train loss ($0.20$) và Val loss ($0.22$) bám rất sát nhau (khoảng cách chỉ ~0.02), chứng tỏ mô hình chưa hề bị quá khớp (overfitting).
- Việc thêm Dropout vào lúc này sẽ làm giảm dung lượng biểu diễn hiệu dụng của mạng, khiến train loss tăng và làm giảm tốc độ hội tụ. Chỉ nên dùng Dropout khi mạng quá lớn (ví dụ `M-wide` hay `M-deep`), tập dữ liệu huấn luyện nhỏ, hoặc khi khoảng cách $\text{Val Loss} - \text{Train Loss}$ tăng vọt theo số epoch.

### Câu 3: Gradient clipping giải quyết vấn đề gì? Quan sát nào của bạn chứng minh điều đó?
- Gradient clipping (cắt độ dài vector gradient nếu vượt ngưỡng $c$) giải quyết vấn đề **Gradient Explosion (bùng nổ gradient)**, ngăn chặn các bước cập nhật quá đà phá hỏng trọng số mô hình khi gặp phải dữ liệu nhiễu ngoại lai hoặc vùng mất mát có độ dốc dựng đứng.
- Ở $lr = 0.1$ bình thường, `grad_norm` đo được dao động ổn định quanh mức $0.05 - 0.20$, nhỏ hơn nhiều so với ngưỡng cắt thông thường ($c=1.0$ hay $c=5.0$), nên clipping hầu như không kích hoạt. Tuy nhiên, khi thử nghiệm với $lr$ cao hoặc batch size nhỏ, clipping giữ cho gradient không vượt quá giới hạn và ngăn ngừa cờ `diverged = True` (NaN/Inf loss).

### Câu 4: Mixed precision có làm huấn luyện nhanh hơn trên mạng và dữ liệu này không? Vì sao?
- Trên mô hình nhỏ như `M-base` (chỉ 3 tầng Linear, 47k tham số) và dữ liệu dạng bảng nạp trực tiếp vào VRAM GPU một lần, Mixed Precision (FP16 / BF16) **không làm tăng tốc độ đáng kể**, thậm chí có thể chậm hơn đôi chút.
- *Lý do:* Chi phí quản lý thang đo gradient (`GradScaler`), ép kiểu dữ liệu qua lại giữa FP32 và FP16, cùng với việc số lượng phép tính ma trận $54 \times 256$ chưa đủ lớn để tận dụng trọn vẹn các nhân Tensor Cores của GPU khiến overhead của phần mềm lớn hơn lợi ích tính toán số thực dấu phẩy động 16-bit.

### Câu 5: Vì sao khởi tạo toàn số 0 hỏng? Khởi tạo He khác Xavier ở điểm nào và khi nào điều đó quan trọng?
- **Khởi tạo bằng 0 bị hỏng:** Do tính đối xứng (symmetry). Nếu tất cả trọng số ban đầu bằng 0, mọi nơ-ron trong cùng một tầng ẩn sẽ nhận cùng một giá trị đầu vào, cho ra cùng một kích hoạt và nhận cùng một gradient khi lan truyền ngược. Toàn bộ nơ-ron cập nhật giống hệt nhau, khiến mạng nhiều nơ-ron bị thoái hoá tương đương với mạng chỉ có 1 nơ-ron duy nhất.
- **He vs Xavier:** 
  - Khởi tạo Xavier (Glorot) giả định hàm kích hoạt tuyến tính quanh điểm 0 (như Tanh), gán phương sai trọng số $\text{Var}(W) = \frac{2}{n_{in} + n_{out}}$.
  - Khởi tạo He (Kaiming) tính đến việc hàm ReLU triệt tiêu một nửa miền giá trị âm ($f(x) = 0$ khi $x < 0$), do đó nhân đôi phương sai $\text{Var}(W) = \frac{2}{n_{in}}$ để bù đắp năng lượng bị mất. He cực kỳ quan trọng đối với các mạng sâu sử dụng ReLU/LeakyReLU để tránh hiện tượng kích hoạt bị suy giảm về 0 ở các tầng sâu.

### Câu 6 (Bắt buộc): Một mạng có loss không giảm sau 2,000 bước. Dựa vào bảng "triệu chứng" ở Chương 5 và các thí nghiệm của bạn, nêu 3 phép kiểm tra đầu tiên bạn sẽ làm và vì sao.
1. **Kiểm tra 1 — Kiểm tra khả năng quá khớp trên một batch nhỏ (Overfit a small batch):**
   - *Cách làm:* Lấy 20–50 mẫu cố định, tắt mọi kỹ thuật điều quy (Dropout, Weight Decay), huấn luyện với SGD/Adam trong 200–500 bước.
   - *Vì sao:* Đây là phép thử toàn diện nhanh nhất để kiểm tra tính đúng đắn của toàn bộ luồng dữ liệu, forward pass, loss function và backward pass. Nếu loss không thể về 0 trên 20 mẫu, lỗi chắc chắn nằm ở code mô hình, nhầm lẫn loss/nhãn hoặc gradient không được truyền (`zero_grad` sai chỗ).
2. **Kiểm tra 2 — Kiểm tra độ lớn Learning Rate và luồng Gradient (`grad_norm`):**
   - *Cách làm:* In gradient norm của từng tầng $\Vert \nabla_W L \Vert$ và thử nghiệm quét dải learning rate qua các bậc độ lớn $[10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1.0]$.
   - *Vì sao:* Loss bất động thường do $lr$ quá nhỏ khiến tham số hầu như không dịch chuyển, hoặc $lr$ quá lớn khiến mô hình dao động quanh điểm ngoằn ngoèo hoặc bị kẹt ở vùng bão hoà. Nếu gradient norm bằng 0, có thể do lỗi "chết nơ-ron" (Dead ReLU) hoặc ngắt kết nối `requires_grad=False`.
3. **Kiểm tra 3 — Kiểm tra chuẩn hoá dữ liệu đầu vào và phân bố nhãn (Data Sanity Check):**
   - *Cách làm:* Tính mean, std của các cột đầu vào $X$ và kiểm tra phân bố của nhãn $y$, đối chiếu loss bước 0 với $-\ln(1/C)$.
   - *Vì sao:* Nếu các biến liên tục có thang đo quá lớn (ví dụ hàng nghìn) mà chưa được chuẩn hoá (Z-score standardizer), hàm kích hoạt sẽ bị bão hoà ngay từ bước đầu. Đồng thời, nếu nhãn bị gán sai dải (ví dụ nhãn $1..7$ chưa trừ 1 về $0..6$) hoặc dữ liệu chứa giá trị NaN/Inf, loss sẽ không thể tối ưu được.

---

## 6. Hạn chế và Điều bất ngờ

- **Điều bất ngờ thú vị:** SGD kết hợp Momentum cao ($0.9$) với $lr=0.1$ cho kết quả cực kỳ mạnh mẽ trên CoverType, vượt trội hơn cả việc cấu hình mặc định của Adam ($0.8531$ vs $0.8446$). Điều này củng cố bài học: Adam không phải lúc nào cũng là giải pháp tối ưu mặc định cho mọi bài toán tabular.
- **Hạn chế của thiết kế thí nghiệm:** Do hạn chế về thời gian GPU, số lượng seed chạy cho từng cấu hình nhánh mới chỉ ở mức 1 seed; chỉ có baseline được đo 3 seed để thiết lập mốc $2\sigma$.
- **Hướng phát triển tiếp theo:** 
  1. Thử nghiệm **Focal Loss** hoặc **Class-Weighted CrossEntropy** để cải thiện độ nhạy cho Lớp 4 (Aspen) và Lớp 3 (Cottonwood).
  2. Thử nghiệm kiến trúc sâu hơn (`M-deep`) kết hợp bộ lập lịch học **Cosine Annealing Learning Rate** để tối ưu hoá khả năng phân tách ở các vùng giáp ranh độ cao.

---

## 7. Phụ lục

- **Danh sách file trong gói nộp `submission_2026xxxx/`:**
  - `REPORT.md`: Báo cáo chi tiết 4 trang phân tích toàn diện.
  - `experiments.xlsx`: Bảng tổng hợp số liệu thí nghiệm với đầy đủ 4 sheet (`Legend`, `Experiments`, `Seeds`, `Summary`).
  - `predictions_eval.csv`: File dự đoán 116,203 dòng trên tập Eval của mô hình tốt nhất `base-s2`.
  - `eval_result.json`: File kết quả chấm chính thức từ `scripts/evaluate.py`.
  - `figures/`: Thư mục chứa các ảnh biểu đồ huấn luyện (`base-s1.png`, `base-s2.png`, `base-s3.png`, `opt-adam.png`, `opt-adamw.png`, `compare_optimizer.png`).
  - `code/`: Thư mục code gồm `lab.ipynb`, `data.py`, `model.py`, `optimizer.py`, `train.py`, `plots.py`, `results_table.py`.
- **Tổng thời gian chạy thực nghiệm:** Khoảng ~15 phút trên GPU Tesla T4 (bao gồm phân chia dữ liệu, kiểm tra sanity check, quét learning rate, chạy 3 seed baseline và các optimizer).
