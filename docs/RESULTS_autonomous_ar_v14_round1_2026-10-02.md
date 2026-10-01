# V14 vòng1: chưa đạt ngưỡng -10%

Hai run H264 có đúng1000 bước,208 clip và2080 dòng ghép cặp, cùng codec thật,
frozen r2plus1d_18 và nămQP. Source/config/checkpoint/chỉ số đã kiểm toán độc lập.

| Công thức | BD-rate Top1 | BD-accuracy | Top1 QP xấu nhất |
|---|---:|---:|---:|
| n3 dual khởi tạo10 | -1,7594% | +0,3187 điểm | -1,4423 điểm tạiQP30 |
| n4 regret từng clip + margin/KL | +2,9879% | -1,1974 điểm | +0,9615 điểm tạiQP30 |

n3 thêm17,04–32,85% bit tại cùngQP và trượt guard QP30. n4 giữ/tăng Top1 ở mọiQP
nhưng thêm9,00–15,76% bit; lợi ích tác vụ không bù được giá bit. Chênh lệch hai
công thức là thăm dò vì dùng seed khác nhau. Không công thức nào được xác nhận.

Vòng kế tiếp đã chuẩn bị: n6 bù chuyển động proxy, n7 giảm duy nhất mu_init10->3.
Hai ứng viên cùng seed302001/source500/ngân sách1000 với n3, giữ engine/model.
Người dùng đã duyệt hai gói mới, sau đó yêu cầu dừng sau lô này.
Không tạo thêm ứng viên hay lịch tự kiểm tra. Mục tiêu/các ngưỡng không đổi, chưa có M_best mới.

802 clip dành riêng khỏi tìm kiếm nhưng từng thuộc validation dùng trước đây;
không coi đây là quần thể kiểm chứng mới. Bằng chứng nằm ở results/autonomous_ar/
và trạng thái/cây giả thuyết trong ops/autonomous_ar/,docs/autonomous_ar_20261001/.

Theo chỉ đạo dừng: n6/n7 đã gửi thành công và được xác minh đang chạy.
Notebook hiện có tự hoàn tất train/eval/xuất báo cáo trên Kaggle. Lịch heartbeat đã xóa;
goal tự tiếp tục được tạm dừng, không tạo thử nghiệm hay lượt thu kết quả tự động mới.
Tại thời điểm dừng chưa có chỉ số n6/n7; không tuyên bố đạt ngưỡng -10%.
