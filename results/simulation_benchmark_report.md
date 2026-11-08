# Báo Cáo Thực Nghiệm Benchmark Thống Kê Toàn Diện & Tối Ưu Siêu Tham Số

> **Thời gian khởi tạo:** 2026-11-07 21:30:42  
> **Thiết kế phân lập Seed:** Tuning Seeds = `[1000, 1001, 1002]` (HPO trung bình) | Evaluation Seeds = `30` seeds độc lập  
> **Cấu trúc phân chia thời gian:** Train 84 tháng, Validation 12 tháng, Test 24 tháng (Tỷ lệ 7:1:2)  

---

## 1. Bảng Đặc Tả Toán Học 6 Kịch Bản Mô Phỏng (Scenario DGP Specifications)

| Kịch bản | Công thức Data Generating Process (DGP) | Tham số Hiệu chỉnh (Calibrated Parameters) | Mục tiêu Kiểm thử (Research Hypothesis) |
|:---|:---|:---|:---|
| **S1: Calibrated Base (Trend + Seasonal + AR, bootstrap noise)** | `$y_t=\max(0,m_t+u_t),\ m_t=a+bt+\sum_{k\le K}[c_k\sin\tfrac{2\pi kt}{P}+d_k\cos\tfrac{2\pi kt}{P}],\ u_t=\sum_{j\le p}\phi_j u_{t-j}+k\sigma s_t\varepsilon_t,\ \varepsilon_t\sim\hat F_{\text{resid}}$` | • <b>season_period_P</b>: 24<br>• <b>n_harmonics_K (BIC)</b>: 11<br>• <b>trend_slope</b>: -247.3 / tháng<br>• <b>ar_order_p (BIC)</b>: 6<br>• <b>phi</b>: [0.305, -0.265, -0.131, -0.289, 0.238, -0.2]<br>• <b>ar_spectral_radius</b>: 0.833<br>• <b>sigma_innov</b>: 35,429 (x noise_scale=1.0)<br>• <b>R2_deterministic / R2_total (calib)</b>: 0.63 / 0.77<br>• <b>calibration_sample</b>: y[0:120] | Kịch bản nền tái tạo cấu trúc trend/mùa vụ/tự tương quan của dữ liệu thật; đo năng lực khai thác cấu trúc động khi nhiễu có hình dạng thực tế. |
| **S2: Base + Seasonal Heteroscedasticity** | `$\text{Base},\ s_t=h_{\text{month}(t)}^{\gamma}$` | • <b>gamma (hetero_power)</b>: 1.0<br>• <b>max/min scale ratio</b>: 5.5 | Kiểm tra khả năng co giãn độ rộng khoảng phân vị theo tháng. |
| **S3: Base + Heavy-tailed Innovations (Student-t)** | `$\text{Base},\ \varepsilon_t\sim t_\nu\sqrt{(\nu-2)/\nu}$` | • <b>nu (t_df)</b>: 3.0 | Kiểm tra độ vững của phân vị đuôi (P5/P95) và chi phí tồn kho khi nhiễu đuôi dày. |
| **S4: Base + Level Shift in Test Period** | `$\text{Base},\ m_t\mathrel{+}=\delta\,\mathbb 1[t\ge\tau]$` | • <b>|delta| / mean</b>: (0.15, 0.25)<br>• <b>tau / n</b>: (0.8, 0.9)<br>• <b>sign</b>: ngẫu nhiên ± | Concept drift xảy ra trong giai đoạn test; đo khả năng thích ứng (vd. RevIN) khi mức nhu cầu đổi đột ngột. |
| **S5: Base + Trend Break** | `$\text{Base},\ m_t\mathrel{+}=\Delta_b\max(0,t-\tau)$` | • <b>|Delta_b| / mean per month</b>: (0.004, 0.008)<br>• <b>tau / n</b>: (0.7, 0.8)<br>• <b>sign</b>: ngẫu nhiên ± | Gãy trend quanh ranh giới train/val: mô hình phải ngoại suy độ dốc mới. |
| **S6: Base + Lagged Exogenous Driver (Promotion)** | `$\text{Base},\ m_t\mathrel{+}=\beta k\sigma\,\tilde g(z_{t-\ell}),\ \tilde g=\text{std}(\tanh z)$` | • <b>beta (exog_beta)</b>: 1.0<br>• <b>lag</b>: 1<br>• <b>driver</b>: PromotionAmount khử trend tuyến tính, chuẩn hóa median/IQR<br>• <b>share of innovation variance</b>: 0.5 | Đo lợi ích của mixing đa biến: chỉ mô hình nhìn thấy biến ngoại sinh trễ mới khai thác được phần tín hiệu này. |

---

## 2. Bảng Siêu Tham Số Tối Ưu Bền Vững Tìm Được Qua Optuna HPO (Trung bình trên Tuning Seeds [1000, 1001, 1002])

| Kịch bản | Mô hình | Lookback | Batch Size | Learning Rate | Hidden Size | N-Layers | N-Stacks/Blocks | Mean Val Pinball | Best Trial |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| S1 | **TSMIXER** | 12 | 8 | 4.34e-03 | 256 | 2 | - | 0.1947 | #21 |
| S1 | **NBEATS** | 24 | 16 | 7.38e-04 | 384 | 2 | 3/4 | 0.0411 | #21 |
| S1 | **NHITS** | 24 | 16 | 3.46e-04 | 192 | 2 | 2/3 | 0.0437 | #24 |
| S2 | **TSMIXER** | 9 | 16 | 5.61e-03 | 448 | 4 | - | 0.1843 | #29 |
| S2 | **NBEATS** | 9 | 16 | 1.69e-03 | 448 | 4 | 2/2 | 0.0398 | #31 |
| S2 | **NHITS** | 9 | 8 | 2.69e-03 | 512 | 4 | 1/2 | 0.0415 | #48 |
| S3 | **TSMIXER** | 12 | 16 | 8.67e-03 | 192 | 4 | - | 0.1020 | #12 |
| S3 | **NBEATS** | 18 | 8 | 3.65e-03 | 448 | 3 | 2/4 | 0.0215 | #32 |
| S3 | **NHITS** | 18 | 8 | 9.41e-03 | 384 | 4 | 2/1 | 0.0233 | #4 |
| S4 | **TSMIXER** | 12 | 8 | 4.34e-03 | 256 | 2 | - | 0.1947 | #21 |
| S4 | **NBEATS** | 24 | 16 | 7.38e-04 | 384 | 2 | 3/4 | 0.0411 | #21 |
| S4 | **NHITS** | 24 | 16 | 3.46e-04 | 192 | 2 | 2/3 | 0.0437 | #24 |
| S5 | **TSMIXER** | 12 | 8 | 4.34e-03 | 256 | 2 | - | 0.1938 | #21 |
| S5 | **NBEATS** | 24 | 16 | 7.38e-04 | 384 | 2 | 3/4 | 0.0410 | #21 |
| S5 | **NHITS** | 24 | 32 | 1.48e-04 | 256 | 3 | 2/4 | 0.0447 | #42 |
| S6 | **TSMIXER** | 12 | 8 | 2.72e-03 | 256 | 2 | - | 0.1875 | #39 |
| S6 | **NBEATS** | 24 | 8 | 3.23e-04 | 384 | 2 | 3/4 | 0.0513 | #49 |
| S6 | **NHITS** | 24 | 32 | 6.01e-03 | 64 | 3 | 2/4 | 0.0533 | #41 |

---

## 3. Bảng Tổng Hợp Skill Scores So Với Oracle (Trung Bình kèm Khoảng Tin Cậy 95% qua 30 Seeds)

> **Ý nghĩa chỉ số Skill Score so với Oracle:**  
> - **Pinball Loss Gap ($\Delta L = \text{Loss}_{\text{model}} - \text{Loss}_{\text{oracle}}$)**: Khoảng cách tiệm cận đến cận dưới lý thuyết hoàn hảo của phân phối.  
> - **Relative Pinball Efficiency ($RPE = \frac{\text{Loss}_{\text{oracle}}}{\text{Loss}_{\text{model}}} \times 100\%$)**: Tỷ lệ phần trăm hiệu suất đạt được so với Oracle (100% = ngang ngửa Oracle).  
> - **Cost Gap ($\Delta TC = \text{TC}_{\text{model}} - \text{TC}_{\text{oracle}}$)**: Chi phí tổn thất do bất định so với Oracle.

| Kịch bản | Mô hình | Test Pinball Loss (Mean ± 95% CI) | Pinball Loss Gap ($\Delta L$) | Relative Pinball Efficiency (%) | Chi Phí Tồn Kho (VND) (Mean ± 95% CI) | Cost Gap ($\Delta TC$) | Relative Cost Efficiency (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| S1 | ORACLE (DGP) | 7,295.3 [6,540.0, 8,050.6] | 0.0 | 100.00% | 6,062,321 [5,288,407, 6,836,236] | 0 | 100.00% |
| S1 | **TSMIXER** | 14,043.9 [13,032.0, 15,055.7] | +6,748.6 | 52.05% | 11,012,221 [10,111,938, 11,912,504] | +4,949,900 | 55.10% |
| S1 | **NBEATS** | 13,836.0 [12,765.9, 14,906.1] | +6,540.7 | 53.54% | 11,498,930 [10,338,553, 12,659,307] | +5,436,609 | 54.95% |
| S1 | **NHITS** | 13,060.6 [12,132.9, 13,988.4] | +5,765.3 | 56.50% | 10,775,645 [9,775,509, 11,775,781] | +4,713,323 | 57.65% |
| S2 | ORACLE (DGP) | 6,417.2 [5,786.7, 7,047.7] | 0.0 | 100.00% | 5,369,851 [4,741,258, 5,998,443] | 0 | 100.00% |
| S2 | **TSMIXER** | 13,580.6 [12,645.5, 14,515.7] | +7,163.4 | 47.83% | 10,900,432 [9,898,724, 11,902,140] | +5,530,582 | 51.99% |
| S2 | **NBEATS** | 14,773.0 [13,484.4, 16,061.6] | +8,355.8 | 44.70% | 12,378,968 [11,032,021, 13,725,915] | +7,009,117 | 45.45% |
| S2 | **NHITS** | 15,033.5 [13,655.4, 16,411.6] | +8,616.3 | 43.88% | 12,126,261 [10,782,756, 13,469,766] | +6,756,410 | 46.12% |
| S3 | ORACLE (DGP) | 6,536.3 [6,056.4, 7,016.1] | 0.0 | 100.00% | 5,363,835 [4,941,924, 5,785,747] | 0 | 100.00% |
| S3 | **TSMIXER** | 12,855.3 [11,842.0, 13,868.6] | +6,319.0 | 51.95% | 10,131,401 [9,016,888, 11,245,915] | +4,767,566 | 55.21% |
| S3 | **NBEATS** | 12,524.9 [11,651.2, 13,398.5] | +5,988.6 | 52.72% | 9,790,171 [8,842,706, 10,737,637] | +4,426,336 | 56.34% |
| S3 | **NHITS** | 14,319.3 [13,141.5, 15,497.0] | +7,783.0 | 46.82% | 11,512,432 [10,058,757, 12,966,108] | +6,148,597 | 50.36% |
| S4 | ORACLE (DGP) | 7,295.3 [6,540.0, 8,050.6] | 0.0 | 100.00% | 6,062,951 [5,288,113, 6,837,789] | 0 | 100.00% |
| S4 | **TSMIXER** | 17,070.3 [15,765.4, 18,375.2] | +9,775.0 | 43.55% | 13,419,488 [12,232,814, 14,606,162] | +7,356,536 | 46.47% |
| S4 | **NBEATS** | 21,571.4 [19,440.8, 23,702.0] | +14,276.1 | 35.22% | 17,935,313 [14,620,858, 21,249,769] | +11,872,362 | 39.09% |
| S4 | **NHITS** | 19,288.3 [17,708.2, 20,868.5] | +11,993.0 | 38.76% | 15,887,081 [13,569,308, 18,204,854] | +9,824,130 | 41.11% |
| S5 | ORACLE (DGP) | 7,295.3 [6,540.0, 8,050.6] | 0.0 | 100.00% | 6,062,794 [5,288,306, 6,837,282] | 0 | 100.00% |
| S5 | **TSMIXER** | 14,589.3 [13,558.1, 15,620.5] | +7,294.0 | 50.23% | 11,339,888 [10,360,330, 12,319,447] | +5,277,094 | 53.75% |
| S5 | **NBEATS** | 16,254.5 [14,863.9, 17,645.1] | +8,959.2 | 45.68% | 13,475,731 [11,593,246, 15,358,216] | +7,412,937 | 48.22% |
| S5 | **NHITS** | 14,854.4 [13,701.7, 16,007.2] | +7,559.2 | 49.75% | 11,769,446 [10,624,420, 12,914,471] | +5,706,652 | 52.23% |
| S6 | ORACLE (DGP) | 7,295.3 [6,540.0, 8,050.6] | 0.0 | 100.00% | 6,069,816 [5,295,903, 6,843,729] | 0 | 100.00% |
| S6 | **TSMIXER** | 16,646.9 [15,327.8, 17,965.9] | +9,351.6 | 44.04% | 14,181,656 [12,626,846, 15,736,466] | +8,111,840 | 44.10% |
| S6 | **NBEATS** | 18,531.2 [17,355.7, 19,706.8] | +11,236.0 | 39.76% | 16,369,814 [14,783,186, 17,956,441] | +10,299,998 | 39.11% |
| S6 | **NHITS** | 17,892.5 [17,135.3, 18,649.7] | +10,597.2 | 40.84% | 16,303,469 [14,827,934, 17,779,004] | +10,233,653 | 39.25% |

---

## 4. Bảng Kiểm Định Giả Thuyết Thống Kê So Sánh Bắt Cặp (Wilcoxon Signed-Rank Test qua 30 Seeds)

> **Quy ước mức ý nghĩa:**  
> - Giả thuyết $H_0$: Không có sự khác biệt có ý nghĩa giữa 2 mô hình trên cùng realization dữ liệu.  
> - Ký hiệu: `*** p < 0.001`, `** p < 0.01`, `* p < 0.05`, `ns: p >= 0.05 (Chấp nhận H0)`.  
> - **Tỷ lệ thắng (Win Rate %)**: Tỷ lệ phần trăm các seed mà Model 1 đạt loss thấp hơn Model 2.

| Kịch bản | Cặp So Sánh | Chỉ Số | Chênh Lệch Trung Bình (Mean Diff) | Tỷ Lệ Thắng (%) | Wilcoxon Statistic ($W$) | Wilcoxon $p$-value | Kết Luận Thống Kê |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| S1 | **TSMIXER vs NBEATS** | Pinball Loss | 207.8 | 43.3% | 196.0 | 0.4645 (ns) | Tương đương (Chấp nhận $H_0$) |
| S1 | **TSMIXER vs NBEATS** | Total Cost | -486,708.8 | 56.7% | 209.0 | 0.6408 (ns) | Tương đương (Chấp nhận $H_0$) |
| S1 | **TSMIXER vs NHITS** | Pinball Loss | 983.2 | 30.0% | 135.0 | 0.0449 (*) | Khác biệt có ý nghĩa |
| S1 | **TSMIXER vs NHITS** | Total Cost | 236,576.3 | 43.3% | 204.0 | 0.5699 (ns) | Tương đương (Chấp nhận $H_0$) |
| S1 | **NBEATS vs NHITS** | Pinball Loss | 775.4 | 33.3% | 138.0 | 0.0523 (ns) | Tương đương (Chấp nhận $H_0$) |
| S1 | **NBEATS vs NHITS** | Total Cost | 723,285.1 | 43.3% | 173.0 | 0.2286 (ns) | Tương đương (Chấp nhận $H_0$) |
| S2 | **TSMIXER vs NBEATS** | Pinball Loss | -1,192.4 | 56.7% | 160.0 | 0.1403 (ns) | Tương đương (Chấp nhận $H_0$) |
| S2 | **TSMIXER vs NBEATS** | Total Cost | -1,478,535.7 | 56.7% | 137.0 | 0.0497 (*) | Khác biệt có ý nghĩa |
| S2 | **TSMIXER vs NHITS** | Pinball Loss | -1,452.9 | 60.0% | 158.0 | 0.1294 (ns) | Tương đương (Chấp nhận $H_0$) |
| S2 | **TSMIXER vs NHITS** | Total Cost | -1,225,828.4 | 56.7% | 175.0 | 0.2449 (ns) | Tương đương (Chấp nhận $H_0$) |
| S2 | **NBEATS vs NHITS** | Pinball Loss | -260.5 | 66.7% | 180.0 | 0.2894 (ns) | Tương đương (Chấp nhận $H_0$) |
| S2 | **NBEATS vs NHITS** | Total Cost | 252,707.4 | 53.3% | 223.0 | 0.8553 (ns) | Tương đương (Chấp nhận $H_0$) |
| S3 | **TSMIXER vs NBEATS** | Pinball Loss | 330.4 | 43.3% | 190.0 | 0.3931 (ns) | Tương đương (Chấp nhận $H_0$) |
| S3 | **TSMIXER vs NBEATS** | Total Cost | 341,229.9 | 46.7% | 175.0 | 0.2449 (ns) | Tương đương (Chấp nhận $H_0$) |
| S3 | **TSMIXER vs NHITS** | Pinball Loss | -1,464.0 | 76.7% | 80.0 | 0.0011 (**) | Khác biệt có ý nghĩa |
| S3 | **TSMIXER vs NHITS** | Total Cost | -1,381,031.1 | 63.3% | 142.0 | 0.0636 (ns) | Tương đương (Chấp nhận $H_0$) |
| S3 | **NBEATS vs NHITS** | Pinball Loss | -1,794.4 | 70.0% | 70.0 | 4.60e-04 (***) | Khác biệt có ý nghĩa |
| S3 | **NBEATS vs NHITS** | Total Cost | -1,722,261.0 | 66.7% | 107.0 | 0.0087 (**) | Khác biệt có ý nghĩa |
| S4 | **TSMIXER vs NBEATS** | Pinball Loss | -4,501.1 | 86.7% | 25.0 | 1.68e-06 (***) | Khác biệt có ý nghĩa |
| S4 | **TSMIXER vs NBEATS** | Total Cost | -4,515,825.9 | 70.0% | 88.0 | 0.0022 (**) | Khác biệt có ý nghĩa |
| S4 | **TSMIXER vs NHITS** | Pinball Loss | -2,218.0 | 56.7% | 122.0 | 0.0221 (*) | Khác biệt có ý nghĩa |
| S4 | **TSMIXER vs NHITS** | Total Cost | -2,467,593.4 | 66.7% | 119.0 | 0.0185 (*) | Khác biệt có ý nghĩa |
| S4 | **NBEATS vs NHITS** | Pinball Loss | 2,283.1 | 36.7% | 122.0 | 0.0221 (*) | Khác biệt có ý nghĩa |
| S4 | **NBEATS vs NHITS** | Total Cost | 2,048,232.4 | 46.7% | 181.0 | 0.2988 (ns) | Tương đương (Chấp nhận $H_0$) |
| S5 | **TSMIXER vs NBEATS** | Pinball Loss | -1,665.2 | 70.0% | 119.0 | 0.0185 (*) | Khác biệt có ý nghĩa |
| S5 | **TSMIXER vs NBEATS** | Total Cost | -2,135,843.0 | 53.3% | 136.0 | 0.0473 (*) | Khác biệt có ý nghĩa |
| S5 | **TSMIXER vs NHITS** | Pinball Loss | -265.1 | 56.7% | 208.0 | 0.6263 (ns) | Tương đương (Chấp nhận $H_0$) |
| S5 | **TSMIXER vs NHITS** | Total Cost | -429,557.2 | 56.7% | 197.0 | 0.4771 (ns) | Tương đương (Chấp nhận $H_0$) |
| S5 | **NBEATS vs NHITS** | Pinball Loss | 1,400.1 | 33.3% | 105.0 | 0.0076 (**) | Khác biệt có ý nghĩa |
| S5 | **NBEATS vs NHITS** | Total Cost | 1,706,285.8 | 43.3% | 145.0 | 0.0732 (ns) | Tương đương (Chấp nhận $H_0$) |
| S6 | **TSMIXER vs NBEATS** | Pinball Loss | -1,884.4 | 70.0% | 107.0 | 0.0087 (**) | Khác biệt có ý nghĩa |
| S6 | **TSMIXER vs NBEATS** | Total Cost | -2,188,158.0 | 60.0% | 120.0 | 0.0197 (*) | Khác biệt có ý nghĩa |
| S6 | **TSMIXER vs NHITS** | Pinball Loss | -1,245.6 | 70.0% | 138.0 | 0.0523 (ns) | Tương đương (Chấp nhận $H_0$) |
| S6 | **TSMIXER vs NHITS** | Total Cost | -2,121,813.1 | 70.0% | 136.0 | 0.0473 (*) | Khác biệt có ý nghĩa |
| S6 | **NBEATS vs NHITS** | Pinball Loss | 638.7 | 46.7% | 194.0 | 0.4400 (ns) | Tương đương (Chấp nhận $H_0$) |
| S6 | **NBEATS vs NHITS** | Total Cost | 66,344.9 | 46.7% | 232.0 | 1.0000 (ns) | Tương đương (Chấp nhận $H_0$) |

---

## 5. Tóm Tắt Kết Quả (sinh tự động từ số liệu, cần tác giả diễn giải thêm)

1. **S1** — Thứ hạng Pinball: NHITS (13,061, RPE 56.5%) < NBEATS (13,836, RPE 53.5%) < TSMIXER (14,044, RPE 52.1%). Mô hình tốt nhất: **NHITS**.  
   Khác biệt có ý nghĩa (Wilcoxon, p<0.05): NHITS < TSMIXER (0.0449 (*)).
2. **S2** — Thứ hạng Pinball: TSMIXER (13,581, RPE 47.8%) < NBEATS (14,773, RPE 44.7%) < NHITS (15,034, RPE 43.9%). Mô hình tốt nhất: **TSMIXER**.  
   Khác biệt có ý nghĩa (Wilcoxon, p<0.05): không có.
3. **S3** — Thứ hạng Pinball: NBEATS (12,525, RPE 52.7%) < TSMIXER (12,855, RPE 51.9%) < NHITS (14,319, RPE 46.8%). Mô hình tốt nhất: **NBEATS**.  
   Khác biệt có ý nghĩa (Wilcoxon, p<0.05): TSMIXER < NHITS (0.0011 (**)), NBEATS < NHITS (4.60e-04 (***)).
4. **S4** — Thứ hạng Pinball: TSMIXER (17,070, RPE 43.5%) < NHITS (19,288, RPE 38.8%) < NBEATS (21,571, RPE 35.2%). Mô hình tốt nhất: **TSMIXER**.  
   Khác biệt có ý nghĩa (Wilcoxon, p<0.05): TSMIXER < NBEATS (1.68e-06 (***)), TSMIXER < NHITS (0.0221 (*)), NHITS < NBEATS (0.0221 (*)).
5. **S5** — Thứ hạng Pinball: TSMIXER (14,589, RPE 50.2%) < NHITS (14,854, RPE 49.7%) < NBEATS (16,255, RPE 45.7%). Mô hình tốt nhất: **TSMIXER**.  
   Khác biệt có ý nghĩa (Wilcoxon, p<0.05): TSMIXER < NBEATS (0.0185 (*)), NHITS < NBEATS (0.0076 (**)).
6. **S6** — Thứ hạng Pinball: TSMIXER (16,647, RPE 44.0%) < NHITS (17,893, RPE 40.8%) < NBEATS (18,531, RPE 39.8%). Mô hình tốt nhất: **TSMIXER**.  
   Khác biệt có ý nghĩa (Wilcoxon, p<0.05): TSMIXER < NBEATS (0.0087 (**)).

---
*Báo cáo được khởi tạo tự động bởi `run_simulation_benchmark.py` phục vụ công bố khoa học.*