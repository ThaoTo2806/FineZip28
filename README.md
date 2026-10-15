# FineZip và Adaptive Context

## Tổng quan

Repository này nghiên cứu nén văn bản không mất dữ liệu bằng mô hình ngôn ngữ nhân quả. FineZip dự đoán token bằng GPT-2, lưu hạng của token thật, sau đó nén chuỗi rank bằng bzip2. Phần mở rộng Adaptive Context dùng cây nhị phân trên các vùng 512 token để chọn chunk dài 32, 64, 128, 256 hoặc 512 token.

Ngoài Adaptive Context, thí nghiệm còn đánh giá context cố định, LoRA/online memorization, các chính sách dựa trên đặc trưng văn bản và các mục tiêu DP khác nhau.

## Dynamic Context của bài gốc và cải tiến Adaptive

Trong FineZip gốc, Dynamic Context Window không phải là việc tự chọn độ dài
chunk cho từng vùng. Ý tưởng của bài gốc là chọn một context size cố định
`L` cho một lần chạy, sau đó token trong mỗi chunk chỉ dùng các token trước đó
trong cùng chunk. Các chunk độc lập nên có thể xử lý theo batch thay vì dùng
sliding window tuần tự.

Adaptive Context là phần cải tiến của repository này: mỗi vùng 512 token được
đặt trên một cây nhị phân, và tree DP hoặc FeaturePolicy chọn `L` riêng cho
từng vùng. Layout được ghi vào file nén để decoder tái tạo đúng cấu trúc.
Vì vậy Fixed Context là baseline của Dynamic Context gốc, còn Adaptive Context
là đóng góp mở rộng được đánh giá trong Giai đoạn 4–5.

## Mục tiêu và câu hỏi nghiên cứu

1. Adaptive Context có giữ được tính lossless và giảm chi phí giải nén so với Fixed Context hay không?
2. Tối ưu theo bit dự đoán có phù hợp với file rank+bzip2 hay không?
3. Tối ưu theo chi phí giải nén gần $L^2$ có cải thiện đường Pareto hay không?
4. Entropy, mức lặp và khả năng dự đoán của LLM có đủ để xây dựng FeaturePolicy rẻ hay không?
5. LoRA có đem lại lợi ích ròng sau khi cộng kích thước adapter hay không?
6. Adaptive có hiệu quả hơn trên dữ liệu không đồng nhất hay không?

## Môi trường và dữ liệu

Các phép đo chính dùng GPT-2 trên GPU Tesla T4 của Kaggle. Dataset chính là lát đầu 1 MB của enwik8:

- Dữ liệu gốc: `1,000,000 bytes`.
- Sau giải mã UTF-8: `995,768 characters`.
- Token GPT-2: `286,273`.

Các ratio trong README dùng mẫu số 1.000.000 bytes. Codec v3 lưu kế hoạch batch hiệu dụng, fingerprint mô hình và thông tin môi trường trong header.

## Kết quả theo giai đoạn

### Giai đoạn 1: Tái lập và chuẩn bị

- Tái lập repository FineZip và pipeline suy luận GPT-2.
- Kiểm tra tokenizer, GPU, model loading, dependency và dữ liệu.
- Xác nhận module arithmetic coding trong repository chưa phải một pipeline LLMZip hoàn chỉnh; pipeline được đánh giá chính ở đây là rank+bzip2.
- Thêm script `update/prepare_data.py` để tạo lát enwik8 1 MB và dataset hỗn hợp
  wiki/code/CSV cho các thí nghiệm tái lập.

### Giai đoạn 2: Baseline và sửa số liệu

- Đối chiếu gzip, bzip2, LLMZip-lite và FineZip.
- Sửa cách tính byte của LLMZip-lite, quy ước ratio, cặp so sánh LoRA và chi phí adapter.
- Baseline 1 MB gồm gzip `355,808 B`, bzip2 `281,323 B` và LLMZip-lite khoảng `186,860 B`.
- Batch size được ghi riêng vì so sánh batch 1 với batch 64 có thể tạo ra chênh lệch tốc độ lớn nhưng không phản ánh thuật toán.

### Giai đoạn 3: Fixed Context và nền tảng LoRA

Kết quả Fixed Context trên 1 MB enwik8 bằng codec v3, không escape:

| Context | Kích thước (B) | Ratio | Encode (s) |
| ------: | -------------: | ----: | ---------: |
|      32 |        300,576 | 3.327 |      27.99 |
|      64 |        271,704 | 3.680 |      28.53 |
|     128 |        246,976 | 4.049 |      26.49 |
|     256 |        228,006 | 4.386 |      27.01 |
|     512 |        213,691 | 4.680 |      27.53 |

Tất cả Fixed configuration đều vượt qua `quick_check`.

LoRA làm giảm kích thước rank stream trước khi tính adapter, nhưng adapter lớn hơn nhiều file nén 1 MB. Có hai cách ghi nhận kích thước:

- Toàn bộ thư mục adapter đo trong session: `6,817,985 B`, gồm weights và các file phụ trợ.
- File weights cần cho việc nạp adapter khi giải nén, theo Excel: `3,253,104 B`.

README dùng số thứ hai khi tính chi phí file cần thiết để giải nén; số toàn bộ thư mục vẫn được ghi để tái lập phép đo filesystem.

### Giai đoạn 4: Codec Adaptive Context

Adaptive-DP chọn layout bằng tree DP. Codec v3 khắc phục lỗi v1 do encoder và decoder có thể dùng batch hiệu dụng khác nhau. Header v3 lưu batch plan cho từng context size, fingerprint model và adapter, thông tin môi trường, cùng escape records tùy chọn.

Kết quả v3 trên 1 MB enwik8:

| Policy              | Kích thước (B) | Ratio | Encode (s) |     Decode (s) |  Lossless   |
| ------------------- | -------------: | ----: | ---------: | -------------: | :---------: |
| Adaptive-DP λ=1000  |        216,534 | 4.618 |     201.73 |    12,000.91\* |     Có      |
| Adaptive-DP λ=5000  |        238,128 | 4.199 |     195.81 | Chưa đo đầy đủ | Quick check |
| Adaptive-DP λ=12000 |        258,127 | 3.874 |     200.78 | Chưa đo đầy đủ | Quick check |
| Adaptive-DP λ=40000 |        288,012 | 3.472 |     201.61 | Chưa đo đầy đủ | Quick check |

`*` `12,000.91 s` là phép đo λ=1000 ở session Kaggle trước. Cùng blob được đo lại trong Phiên D: Fixed-512 mất `13,516.68 s`, còn Adaptive-DP-1000 mất `10,833.34 s`. Do thời gian giữa các session biến động, chỉ so sánh hai policy trong cùng session.

Trong cùng Phiên D:

- Fixed-512: `213,691 B`, decode `13,516.68 s`, lossless.
- Adaptive-DP-1000: `216,534 B`, decode `10,833.34 s`, lossless.
- Adaptive lớn hơn `1.33%` về kích thước.
- Adaptive giảm `19.85%` thời gian giải nén.

Escape được kiểm tra như cơ chế portability nhưng làm file phình mạnh khi epsilon được hiệu chuẩn theo nhiễu giữa các batch. Kết quả chính dùng codec v3, `eps=None`, zero escape và batch plan được lưu trong header.

#### Mã giả của pipeline gốc và pipeline cải tiến

Dynamic Context gốc với context cố định:

```text
tokens <- tokenize(text)
for each non-overlapping chunk of length L:
	input <- [BOS] + chunk[:-1]
	rank <- model_rank(input, chunk)
compress all ranks with bzip2
```

Adaptive Context cải tiến:

```text
tokens <- tokenize(text); pad to multiples of 32
for each 512-token region:
	score candidate chunks L in {32, 64, 128, 256, 512}
	choose leaf or split node with tree DP / feature policy
store chosen layout and effective batch plan
encode ranks for chosen leaves; compress ranks with bzip2
decoder reads layout and batch plan, then reconstructs each leaf losslessly
```

Tác động của thay đổi: Adaptive có thể giảm chi phí giải nén bằng cách chọn
chunk ngắn ở vùng khó hoặc ít lợi từ context dài, nhưng làm tăng header/layout
và có thể làm bzip2 kém hiệu quả hơn do rank stream bị phân mảnh. DP cũng phải
đánh đổi thêm chi phí chấm điểm nhiều mức context ở phía encoder.

### Giai đoạn 5: So sánh có kiểm soát

#### 5A. Mục tiêu DP và giả thuyết rank+bzip2

Ba mục tiêu AC, calibrated và rank-proxy được so ở cùng effective context $\bar L$. Kết quả trên 1 MB cho thấy các đường Pareto gần nhau; rank-proxy chỉ tốt hơn nhẹ. Vì vậy chưa có bằng chứng rằng đổi mục tiêu cục bộ sẽ tạo cải thiện lớn. Sự lệch giữa bit dự đoán và rank+bzip2 có tồn tại, nhưng tác động nhỏ trên dataset này.

Thay objective thời gian encode bằng chi phí giải nén xấp xỉ $L^2$ làm policy chọn chunk ngắn hơn, nhưng không tạo ưu thế rõ ràng so với Fixed. Đây là kết quả âm cần giữ lại trong báo cáo.

#### 5B. Dataset không đồng nhất

Dataset hỗn hợp gồm `964,296 bytes` UTF-8 và `425,852` token GPT-2, ghép wiki, Python source và các dòng dạng CSV. Fixed baseline:

| Context | Kích thước (B) | Ratio |
| ------: | -------------: | ----: |
|      32 |        381,518 | 2.528 |
|      64 |        327,262 | 2.947 |
|     128 |        284,646 | 3.388 |
|     256 |        255,050 | 3.781 |
|     512 |        235,345 | 4.097 |

Adaptive ở cùng $\bar L$ gần Fixed nhưng chưa thắng rõ. Feature tốt nhất là bigram repeat với `gain_captured=0.417`, dưới ngưỡng làm việc `0.60`. ThresholdPolicy-repeat vẫn lossless qua quick check và cho `279,355 B` tại $\bar L=147.39$, nhưng chưa đạt chất lượng DP.

#### 5C. Ablation LoRA

LoRA được nạp bằng PEFT và merge trước suy luận. Fingerprint codec có chứa SHA-256 của adapter. Nếu không tính adapter, LoRA giảm file:

- Fixed-512: giảm `1.99%`, từ `213,691 B` xuống `209,447 B`.
- Adaptive-DP-1000: giảm `1.92%`, từ `216,534 B` xuống `212,371 B`.

Nếu tính chi phí file weights cần để giải nén, dùng `3,253,104 B`:

- LoRA Fixed-512: tổng `3,462,551 B`, ratio ròng khoảng `0.289`.
- LoRA Adaptive-DP-1000: tổng `3,465,475 B`, ratio ròng khoảng `0.288`.

Nếu tính toàn bộ thư mục adapter `6,817,985 B`, ratio ròng lần lượt là `0.142` và `0.142`. Cả hai quy ước đều dẫn đến cùng kết luận: LoRA không có lợi ròng cho file 1 MB. Khi báo cáo, cần ghi rõ quy ước kích thước adapter.

#### 5D. Đo thời gian giải nén

Phiên D đo Fixed-512 và Adaptive-DP-1000 trong cùng session, cùng model, cùng GPU và cùng codec. Đây là phép đo chính cho tuyên bố tiết kiệm decode. Không dùng các số decode giữa những session khác nhau để kết luận tốc độ tuyệt đối.

## Những điểm đã sửa: trước và sau

### Trước khi sửa

- v1 giả định batch không ảnh hưởng thứ hạng rank.
- Decoder có thể dùng shape batch khác encoder.
- Logit GPT-2 giữa các batch lệch tới khoảng `0.02` trong môi trường đo.
- Cache DP ban đầu chưa gắn đủ dữ liệu/model/batch context.
- DP dùng proxy thời gian encode trong khi decode autoregressive có xu hướng gần $L^2$.
- LoRA chưa nằm trong fingerprint và kích thước adapter dễ bị bỏ sót.
- README cũ trộn kích thước file gốc với số character sau giải mã UTF-8.

### Sau khi sửa

- v3 lưu batch plan hiệu dụng cho từng context size.
- v3 lưu metadata model, môi trường và fingerprint adapter.
- Decoder từ chối model không tương thích và cảnh báo môi trường khác.
- Escape có thể dùng khi cần portability giữa môi trường số khác nhau.
- DP cache key gồm dữ liệu, model identity và batch context.
- HFLM hỗ trợ nạp, merge LoRA và băm SHA-256 adapter.
- Các phép đo Fixed, Adaptive, mixed-data, FeaturePolicy và LoRA được lưu JSON.

## Điểm mạnh và điểm yếu

### Điểm mạnh

- Codec rank+bzip2 end-to-end lossless đã được triển khai và kiểm tra.
- Adaptive tạo trade-off đo được: file tăng 1.33% nhưng decode nhanh hơn 19.85% so với Fixed-512 trong cùng session.
- Batch plan và fingerprint tăng khả năng tái lập và phát hiện cấu hình sai.
- Fixed baseline dùng cùng codec v3, nên so sánh chặt hơn các số liệu khác pipeline.
- Feature training dùng train/test split, giảm nguy cơ rò rỉ dữ liệu.

### Điểm yếu

- Adaptive chưa thống trị Fixed trên cả kích thước và thời gian decode tại cùng operating point.
- Encode DP gồm chấm điểm cả năm context nên không so trực tiếp với một lượt encode Fixed.
- Decode dao động khoảng 26% giữa các session Kaggle.
- Dataset hỗn hợp là dữ liệu tự tạo có kiểm soát, chưa phải benchmark chuẩn.
- FeaturePolicy chỉ có tín hiệu vừa phải trên dataset hỗn hợp.
- LoRA không có lợi ròng ở quy mô 1 MB vì adapter lớn.
- Chưa decode đầy đủ mọi lambda; nhiều điểm chỉ có quick-check.

## Đối chiếu yêu cầu đề bài

- Source code và hướng dẫn: có codec v3, stage5 tools và README này.
- Baseline gzip/bzip2/LLMZip-lite/FineZip: đã có trong Giai đoạn 2 và Excel.
- Context 32–512: đã chạy Fixed đầy đủ bằng codec v3.
- Adaptive Context: đã chạy DP, lưu layout/batch plan và kiểm tra lossless.
- So sánh ratio, encode, decode và memory: đã có cho các cấu hình chính; decode chính được đo cùng session.
- Ablation LoRA: đã có, gồm cả hai quy ước tính kích thước adapter.
- Ablation chính sách: đã có AC/calibrated/rank-proxy và FeaturePolicy.
- Ablation dữ liệu: đã có lát enwik8 và dataset hỗn hợp.
- Giới hạn và trường hợp Adaptive không hiệu quả: đã được ghi rõ.
- Ý tưởng bài báo, mã giả pipeline gốc/cải tiến, tác động của thay đổi và bảng
  so sánh: có trong các mục Dynamic Context, Mã giả, Kết quả và giới hạn của README.
- Script chuẩn bị dữ liệu và cấu hình thí nghiệm: `update/prepare_data.py`, các
  notebook `notebook/version*.ipynb` và `update/stage5_tools.py`.

## Sản phẩm bàn giao

- Source code có hướng dẫn: codec v3, stage5 tools và README.
- Script chuẩn bị dữ liệu: `update/prepare_data.py`.
- Notebook và cấu hình thí nghiệm: `notebook/version1.ipynb` đến các notebook
  Giai đoạn 3–5.
- Mã giả Dynamic Context gốc và Adaptive Context cải tiến.
- Phân tích tác động thay đổi, bảng kết quả, so sánh Fixed/Adaptive và ablation.

## Artifact tái lập

- `update/adaptive_context_finezip_v3.py`: codec v3.
- `update/adaptive_context_finezip_v2.py`: codec v2 có escape.
- `update/stage5_tools.py`: TreeDP, hiệu chuẩn mục tiêu, feature fitting và report.
- `update/prepare_data.py`: tạo lát enwik8 1 MB và dataset hỗn hợp.
- `adaptive_results_v3_final.json`: Adaptive-DP trên 1 MB enwik8.
- `stage5_session_A.json`: objective comparison và FeaturePolicy.
- `stage5_session_B.json`: dataset không đồng nhất.
- `stage5_session_C.json`: LoRA ablation.
- `stage5_session_D.json`: đo decode cùng session.
- `stage5_analysis_A.npz`: cache score/rank theo context.
- `stage5_decode_cost.json`: thử DP theo chi phí decode.
- `update/FineZipExperiment_Corrected_v6.xlsx`: bảng tổng hợp thí nghiệm mới nhất.

## Tóm tắt trung thực

Dự án hiện có codec Adaptive Context lossless và bộ thí nghiệm có kiểm soát cho Fixed Context, mục tiêu DP, dữ liệu không đồng nhất, FeaturePolicy và LoRA. Trên lát enwik8 1 MB, Adaptive-DP-1000 đánh đổi việc file lớn hơn `1.33%` để giảm `19.85%` thời gian giải nén trong cùng session. Kết quả ủng hộ Adaptive như một cơ chế điều khiển trade-off compression/decode, nhưng chưa chứng minh đây là cải tiến phổ quát so với Fixed Context. Kết quả âm quan trọng là LoRA và FeaturePolicy đã thử chưa đem lại lợi ích ròng ở quy mô dữ liệu này.

## Thành phần FineZip gốc

- `finezip/eval_clean.py`: đánh giá FineZip rank+bzip2.
- `finezip/finetune.py`: script fine-tune LoRA/QLoRA.
- `AC/arithmeticcoding.py`: tiện ích arithmetic coding.
- `AC/eval_ac.py`: đánh giá arithmetic coding với context cố định.
