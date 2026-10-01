# Kết quả V13 task-regret — kiểm toán ngày 01/10/2026

**Quyết định: dừng V13 cho cả H.264 và H.265.** Cả hai không đạt thiết kế đã
đăng ký và không đạt yêu cầu BD-rate ≤ −10% cho từng codec của người dùng.

Hai notebook đã hoàn tất trên Kaggle từ 22/09/2026; đợt này tải và kiểm toán
artifact của các lượt đó. Không chạy lại training, không thay epsilon, dual LR,
loss weight hay checkpoint để tìm kết quả tốt hơn trên màn hình đã xem.

| Codec | Seed | BD-rate theo Top-1 | BD-accuracy (điểm %) | QP giảm Top-1 nhiều nhất | Quyết định |
|---|---:|---:|---:|---:|---|
| H.264 | 292001 | **+3,7335%** | −1,6643 | −5,7692 tại QP35 | Dừng |
| H.265 | 292002 | **+15,7634%** | −11,9478 | −26,4423 tại QP35 | Dừng |

BD-rate được tính riêng `prep+h264 vs h264` và `prep+h265 vs h265`, bằng
Top-1 toàn tập làm trục chất lượng. Không lấy phép so sánh chéo codec hay
trung bình BD-rate của từng clip làm kết quả chính.

## Đúng thiết kế đăng ký

- Code training: commit `35f0bd1c7844ad6965f5b0e148f8bd7869f48d93`.
- Cả hai checkpoint cuối có `epoch=3`, `global_step=12948`; SHA-256 khớp
  manifest gốc. Cấu hình task/model/loss/train khớp YAML V13; codec chỉ khác
  trường `ste_codec` ở nhánh H.265 như đã đăng ký.
- Log xác nhận tải trọng số từ nguồn 500 bước và dùng optimizer mới, LR `1e-5`.
  Old rate duals đều bằng 0; có đúng năm task dual riêng cho từng codec/QP.
- `epsilon=0`, `dual_lr=0.001`, `mu_init=0.1`, `mu_max=10`, `rate_weight=1`.
  Tất cả task dual cuối hữu hạn và trong [0,10].
- Index có đúng 8.632 train / 1.010 val / 1.159 test; membership hash của từng
  split hợp lệ, fingerprint test là `30f083f8520a`.
- Đánh giá checkpoint cuối bằng held-out `r2plus1d_18`, codec thật, QP
  30/35/40/45/50; shard 0/5 với salt `v13-task-regret-screen-v1`.
- Màn hình có đúng 208 clip; fingerprint được tính lại từ các ID và khớp
  `f321b810af2c507c`. Mỗi codec có đủ 2.080 điểm ghép cặp (208 × 5 × 2),
  không thiếu/trùng clip hay QP. Curve tổng hợp khớp các bản ghi từng clip.
- BD-rate/BD-accuracy được tính lại bằng mã metric canonical; khớp số gốc
  trong sai số số học.

## Guardrail theo QP

| QP | H.264 ΔTop-1 (điểm %) | H.265 ΔTop-1 (điểm %) |
|---:|---:|---:|
| 30 | −3,3654 | −17,3077 |
| 35 | −5,7692 | −26,4423 |
| 40 | −3,8462 | −25,0000 |
| 45 | −5,2885 | −12,5000 |
| 50 | −2,4038 | −1,9231 |

Ở cùng QP, V13 giảm số bit khoảng 3,17–8,41% với H.264 và 4,82–20,15% với
H.265, nhưng mức giảm Top-1 làm BD-rate tại cùng chất lượng trở thành dương.
Kết quả này không hỗ trợ mục tiêu máy nhận diện tốt hơn ở cùng bitrate cho
cấu hình learned-AR V13.

## Quyết định và phạm vi

Thiết kế đăng ký yêu cầu đồng thời BD-rate ≤ −5%, BD-accuracy ≥ 0 và không QP
nào mất quá 0,5 điểm Top-1. Cả hai codec trượt cả ba điều kiện. Ngưỡng bổ sung
BD-rate ≤ −10% được người dùng xác nhận ngày 01/10/2026 và được ghi riêng;
không sửa ngưỡng đăng ký ban đầu. Không kích hoạt replication/full-val/test.

Engine và loss hiện có tiếp tục là hạ tầng chung cho OD/AR; đợt này không tạo
pipeline mới hoặc tách mã training. Thiết kế V13 vẫn train hai checkpoint theo
codec, như đã đăng ký. Màn hình AR này không xác nhận một model chung OD/AR.
OD giữ trạng thái freeze trong `RESULTS_generalization_matrix.md`.

Một thiết kế tiếp theo, kể cả tối ưu tham số loss hoặc model chung OD/AR, cần
được đăng ký riêng với kiểm chứng độc lập; không đổi nhãn kết luận V13 đã thất bại.

## Artifact và tái kiểm toán

- [Gói số liệu và lệnh tái kiểm toán](../results/v13_task_regret_screen/README.md).
- [Tổng hợp máy đọc](../results/v13_task_regret_screen/summary.json).
- [H.264 notebook](https://www.kaggle.com/code/qktttttttttt/preupd-v13-task-regret-h264-screen-r1).
- [H.265 notebook](https://www.kaggle.com/code/baoancut/preupd-v13-task-regret-h265-screen-r1).

`ops/report_v13_task_regret.py` kiểm tra split/fingerprint, toàn bộ điểm ghép
cặp, metric, guardrail và checkpoint/cấu hình. Runner hiện có nhúng helper này
vào notebook cùng commit training bất biến; kết quả BD không xác định được
báo là futility stop. Không có thay đổi thuật toán training.

## Kiểm thử mã kiểm toán

29/29 kiểm thử liên quan đã qua trên CPU, gồm loss V13, conditioning codec/QP,
báo cáo sequence BD và kiểm toán màn hình/checkpoint. Các trường hợp thiếu/trùng
clip, sai metric, đổi epsilon, thiếu bước, kế thừa rate dual, sai SHA và chọn
checkpoint best thay final đều bị từ chối. `git diff --check` cũng đã qua.

```powershell
python -m pytest tests/test_v13_task_regret.py tests/test_v13_screen_report.py tests/test_crc_v5.py tests/test_sequence_bd_report.py -q
```
