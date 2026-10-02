# Kết quả V14 cuối — kiểm tra thủ công ngày 2026-10-02

Cả bốn job H264 đã hoàn tất. Hai job cuối n6/n7 được tải và kiểm toán theo yêu cầu
"check ket qua"; lịch tự kiểm tra đã xóa, goal tự tiếp tục vẫn tạm dừng.

Mỗi run dùng codec thật, frozen r2plus1d_18, năm QP30/35/40/45/50, cùng 208 clip
sàng lọc (fingerprint f321b810af2c507c), 2080 dòng ghép cặp và 1000 bước mới từ
source H264 step500. Mã/cấu hình/checkpoint/partition/curve/gate đã đối chiếu độc lập.

| Phiên bản | BD-rate Top1 | BD-accuracy (điểm) | Top1 QP xấu nhất (điểm) |
|---|---:|---:|---:|
| n3: Dual khởi tạo 10 | -1.7594% | +0.3187 | -1.4423 |
| n4: Regret từng clip + margin/KL | +2.9879% | -1.1974 | +0.9615 |
| n6: Bù chuyển động proxy | -0.6846% | +0.4822 | +0.4808 |
| n7: Dual khởi tạo 3 | +2.1869% | -0.4226 | -0.4808 |

Không run nào đạt BD-rate <= -10%. n6 đạt cả guard Top1 từng QP và BD-accuracy >=0,
nhưng chỉ giảm 0.6846% bitrate ở cùng hiệu năng tác vụ. n7 đạt guard mất không quá
0.5 điểm Top1 từng QP nhưng BD-rate +2.1869%, BD-accuracy âm.

Tại cùng QP, n6 tăng bitrate17.51–33.62%; n7 tăng12.49–22.83%. Cải thiện nhận diện
ở cùng QP chưa đủ bù giá bit. So với n3 có cùng source/seed/ngân sách, n6 khắc phục
mất Top1 QP30 nhưng không đạt mục tiêu bitrate; hạ duy nhất mu_init10->3 trong n7
không cải thiện BD-rate. Đây là giới hạn quan sát của các screen này, không suy rộng
thành thất bại của mọi công thức. n4 dùng seed khác nên đối chiếu n4 là thăm dò.

802 clip xác nhận và full1010 chưa được đánh giá cho các ứng viên này vì không run
nào qua gate phát triển. 802 clip chỉ tách khỏi tìm kiếm hiện tại; từng thuộc tập
validation dùng trước đây, không phải quần thể kiểm chứng mới. Chưa có M_best mới.

Bằng chứng từng run nằm tại results/autonomous_ar/<slug>/ gồm local_audit.json,
checkpoint_manifest.json, job_spec.json, summary.json, paired CSV và SHA từng file.
Checkpoint thô nằm trong outputs/autonomous_ar/downloads/ (bỏ qua Git).

Theo yêu cầu dừng: không tạo thử nghiệm mới, không bật lịch kiểm tra, không khôi phục
goal tự tiếp tục. Kết quả n6/n7 được thu đúng một lần theo yêu cầu thủ công hiện tại.
