# Stage 7 — Decode-Aware Budget-Constrained Adaptive FineZip

**Trạng thái:** hoàn thành thực nghiệm Stage 7 (22/11/2026). Đây là phần mở rộng codec FineZip v3, không thay đổi source gốc. Kết quả được tổng hợp từ log Kaggle Cells 1–18 do nhóm cung cấp.

## 1. Mục tiêu và đóng góp

Stage 6A đã hoàn thành full decode 9/9 cấu hình trên enwik8 1 MB nhưng chưa chứng minh Adaptive vượt Fixed ở cùng ngân sách giải nén. Stage 7 nghiên cứu hiệu chuẩn chi phí giải nén thực tế, lựa chọn λ theo từng tệp và sàng lọc layout bằng kích thước rank stream nén BZip2. Phương pháp gọi tắt là BC-DP (Budget-Constrained Dynamic Programming).

## 2. Môi trường và dữ liệu

- Repository: https://github.com/ThaoTo2806/FineZip28; commit `62f09e3105296d62c39e8d95cf8b8775e63f2677`.
- GPT-2 FP32, SDPA, vocabulary 50.257, BOS 50.256; không dùng LoRA, không escape.
- Tesla T4 GPU 0; PyTorch 2.10.0+cu128, CUDA 12.8, Transformers 5.0.0, NumPy 2.0.2; TF32 tắt, matmul precision highest.
- Codec v3: rank uint16 + BZip2, layout và batch plan trong header; context 32/64/128/256/512, batch 8/8/8/4/2.
- Session đo Stage7-Session01.

| Vai trò | Offset enwik8 | Byte | Token | SHA-256 |
|---|---:|---:|---:|---|
| Calibration | 2.000.000 | 100.000 | 26.497 | `0c1f9fa18536c237b18991e06b4a645e3fc741e02358e27965f9f2b9d60a1e8f` |
| Development/validation | 3.000.000 | 100.000 | 27.461 | `8ed483f6ebd5c8b8d601a78af4fd76a5cd7d243d4ab8399745581a5be9630403` |
| Held-out | 4.000.000 | 100.000 | 28.313 | `06e7b14a1f7cdc48bf5142e65fd563123464fecfa8a2ccda75cea47249079651` |

Các lát được chuẩn bị dưới dạng UTF-8 hợp lệ; lossless được xác nhận theo byte của dữ liệu đầu vào sau chuẩn bị. Held-out khác vùng calibration/development nhưng vẫn thuộc cùng corpus enwik8.

## 3. Hiệu chuẩn latency

Đo `_decode_group` trên rank thật, warm-up, 5 lần/context, đảo thứ tự, đồng bộ GPU trước và sau đo. 25/25 lần đo khớp token.

| L | B | Median batch (s) | Median chunk (s) | CV (%) |
|---:|---:|---:|---:|---:|
| 32 | 8 | 0,782312 | 0,097789 | 2,7902 |
| 64 | 8 | 3,139418 | 0,392427 | 4,4136 |
| 128 | 8 | 12,041096 | 1,505137 | 1,2702 |
| 256 | 4 | 24,120105 | 6,030026 | 1,6042 |
| 512 | 2 | 48,545590 | 24,272795 | 1,1314 |

Chi phí cũ `1.02e-4 * L²`; chi phí hiệu chuẩn `9.259336407279944e-5 * L²`; bảng measured dùng median batch chia B. MAPE trên 9 kết quả lịch sử Stage 6A (khác session) lần lượt 16,6348%, 5,8785%, 6,8977%. Fitted quadratic giảm sai số 64,66% so với hệ số cũ; measured không thắng fitted ở phép đối chiếu lịch sử này. Thay hệ số c đơn thuần có thể được bù bởi λ, do đó không coi hiệu chuẩn đơn lẻ là đóng góp thuật toán mới.

## 4. Thuật toán

Với mỗi vùng 512 token, xét cây nhị phân có lá từ 32 đến 512 token. Với nút v có độ dài L, chi phí lá là `C(v) + λ T(L)`, trong đó C là NLL hoặc rank proxy, T là quadratic hiệu chuẩn hoặc measured latency. TreeDP chọn min giữa giữ lá và tổng chi phí hai nút con.

Rank proxy: histogram rank từ calibration ở L=128, clip 4096, smoothing 0,5; rank hiếm có chi phí đuôi log2(rank). Histogram được cố định khi đánh giá trên dữ liệu mới.

```text
BC_DP(tokens, target_fixed_context, frozen_costs):
    pad tokens to multiple of 32
    score and cache NLL and ranks for five context sizes
    budget = predicted latency of fixed reference layout
    for compression_objective in [NLL, Rank]:
        for time_model in [Fitted, Measured]:
            for lambda in [0] + geomspace(0.01, 10000, 121):
                layout = TreeDP(compression_objective, time_model, lambda)
                deduplicate layout by hash
                variable_bytes = len(BZip2(cached ranks)) + packed layout bytes
                predicted_decode = sum(chunk_count[L] * measured_chunk_time[L])
    select feasible layout with minimum variable_bytes,
        breaking size ties by lower predicted_decode
    re-encode chosen layout with codec v3
    return compressed blob
```

Đây là tìm kiếm hữu hạn trên các ứng viên DP, không bảo đảm tối ưu toàn cục. BZip2 trên rank cache và layout là proxy sàng lọc (không tính header cố định); kích thước cuối cùng lấy từ file codec thực tế. Ngân sách latency dự báo chứ không phải SLA được bảo đảm. Chi phí phân tích và tìm kiếm được cộng vào encode của mỗi BC khi tính tổng chi phí.

## 5. Kết quả phát triển

- Calibration: 5 context, 150 cấu hình, 77 layout Adaptive khác nhau, 6/24 phép so sánh ngân sách có size saving dương; tốt nhất 1,1865% (Rank-Old vs Fixed-64).
- Frozen λ trên development: 8/8 encode, Adaptive tăng kích thước 0,55–1,50% nhưng giảm decode dự báo 6,42–15,97%; không có cặp nào thắng cả hai.
- Dense budget search trên development: 488 cấu hình, 158 layout, chỉ 1/4 ngân sách có size saving dương (0,1357% ở Fixed-512), dự báo giảm decode 3,0672%.
- Held-out: 488 cấu hình, 160 layout, search 1,578 s, analysis 14,567 s; 8/8 encode và 16/16 full decode qua các lần lặp. Không điều chỉnh lại quy trình tìm kiếm sau khi xem held-out.

## 6. Kết quả chính trên held-out (100 KB)

| L | Fixed B | BC B | Fixed encode s | BC encode s | Fixed decode mean ± SD (s) | BC decode mean ± SD (s) | n/method |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 27731 | 27622 | 3,1571 | 19,4874 | 174,3666 ± 0,2985 | 169,5271 ± 0,3221 | 3 |
| 128 | 25644 | 25541 | 3,0158 | 19,4614 | 330,3714 ± 0,5423 | 321,6849 ± 0,3404 | 3 |
| 256 | 23921 | 23909 | 3,1164 | 19,4423 | 663,0023 | 653,0258 | 1 |
| 512 | 22794 | 22738 | 3,2415 | 19,4335 | 1342,8103 | 1314,5340 | 1 |

| L | Size saving (%) | Mean paired decode saving (%) | Encode slowdown | Total time saving (%) | Decode peak GPU change (%) | Break-even decodes |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0,3931 | 2,7752 | 6,1725x | -6,4728 | +20,0332 | 4 |
| 128 | 0,4017 | 2,6290 | 6,4531x | -2,3273 | 0 | 2 |
| 256 | 0,0502 | 1,5047 | 6,2388x | -0,9532 | 0 | 2 |
| 512 | 0,2457 | 2,1058 | 5,9953x | +0,8978 | 0 | 1 |

Break-even là số lần giải nén ước lượng để bù chi phí encode tăng thêm, giả định decode time trung bình không đổi và cùng điều kiện. Không bao gồm model tải, cold start hay I/O ngoài phép đo. MAPE latency held-out trên 16 phép đo: 0,514%. BC-64 và BC-128 đều nhanh hơn 3/3 lần; 256/512 chỉ đo một lần nên không có SD. Tất cả 16 phép đo byte-exact, token-exact và SHA-256 PASS (8 artifact khác nhau, không phải 16 dataset độc lập).

## 7. Kết luận và giới hạn

BC-DP cho thấy mức giảm dung lượng 0,0502–0,4017% và giảm decode 1,5047–2,7752% ở 4 ngân sách trên một lát held-out, nhưng encode chậm khoảng 6 lần. BC-64 tăng bộ nhớ GPU giải nén; các cấu hình còn lại không tăng. Chỉ BC-512 giảm tổng thời gian encode + một lần decode tại lần đo hiện có. Không tuyên bố thắng phổ quát hay có ý nghĩa thống kê trên nhiều corpus; đây là bằng chứng bước đầu. So sánh với Fixed ở cùng ngân sách dự báo không thay thế so sánh với Fixed trung gian hoặc Adaptive cũ, random layout, các biến thể ablation trên dữ liệu đa dạng.

## 8. Artifact và bước tiếp theo

- `results/`: latency raw/summary, cost model comparison, sweep, budget search, validation/held-out encode/decode, final summary và paired comparison (CSV/JSON).
- `plots/`: 9 hình (PNG + PDF): latency calibration, historical prediction, calibration Pareto, held-out actual Pareto, improvements, held-out latency prediction, repeat stability, encoding overhead, total time trade-off.
- `code/`: notebook history và/hoặc code thuật toán bổ sung khi đóng gói; giữ nguyên source gốc.
- `artifacts/`: calibration cache `.npz`, frozen rank table `.npy`, layout JSON, `.fzac`, text khôi phục; không có model weights mới.
- Stage 8: giữ frozen search, kiểm chứng trên nhiều bộ dữ liệu độc lập; tái lập Fixed và BC cùng session, full decode/SHA-256, báo cáo cả trường hợp thất bại. Stage 9: đối chứng với DP cũ, random layout, ablation cost/search và đo lặp; Stage 10: tổng hợp paper.
