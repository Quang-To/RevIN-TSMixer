# Báo Cáo Thực Nghiệm Benchmark Thống Kê Toàn Diện & Tối Ưu Siêu Tham Số

> **Thời gian khởi tạo:** 2026-11-08 13:55:30  
> **Thiết kế phân lập Seed:** Tuning Seeds = `[1000, 1001, 1002]` (HPO trung bình) | Evaluation Seeds = `30` seeds độc lập  
> **Cấu trúc phân chia thời gian:** Train 84 tháng, Validation 12 tháng, Test 24 tháng (Tỷ lệ 7:1:2)  

---

## 1. Bảng Đặc Tả Toán Học 6 Kịch Bản Mô Phỏng (Scenario DGP Specifications)

| Kịch bản | Công thức Data Generating Process (DGP) | Tham số Hiệu chỉnh (Calibrated Parameters) | Mục tiêu Kiểm thử (Research Hypothesis) |
|:---|:---|:---|:---|
| **S1: Calibrated Base (Trend + Seasonal + AR, bootstrap noise)** | `$y_t=\max(0,m_t+u_t),\ m_t=a+bt+\sum_{k\le K}[c_k\sin\tfrac{2\pi kt}{P}+d_k\cos\tfrac{2\pi kt}{P}],\ u_t=\sum_{j\le p}\phi_j u_{t-j}+k\sigma_t s_t\varepsilon_t,\ \sigma_t=\sigma_{\text{pre}}\mathbb 1[t<\tau_\sigma]+\sigma_{\text{post}}\mathbb 1[t\ge\tau_\sigma],\ \varepsilon_t\sim\hat F_{\text{resid}}$` | • <b>season_period_P</b>: 24<br>• <b>n_harmonics_K (BIC)</b>: 12<br>• <b>trend_slope</b>: -106.5 / tháng<br>• <b>ar_order_p (BIC)</b>: 5<br>• <b>phi</b>: [0.781, -0.644, 0.32, -0.08, 0.201]<br>• <b>ar_spectral_radius</b>: 0.828<br>• <b>regime_break_tau</b>: 36<br>• <b>phi_pre (trước tau)</b>: []<br>• <b>sigma_innov_pre</b>: 97,822 (x noise_scale=1.0)<br>• <b>sigma_innov_post</b>: 4,833 (x noise_scale=1.0)<br>• <b>R2_deterministic / R2_total (calib)</b>: 0.48 / 0.46<br>• <b>calibration_sample</b>: y[0:120] | Kịch bản nền tái tạo cấu trúc trend/mùa vụ/tự tương quan của dữ liệu thật; đo năng lực khai thác cấu trúc động khi nhiễu có hình dạng thực tế. |
| **S2: Base + Seasonal Heteroscedasticity** | `$\text{Base},\ s_t=h_{\text{month}(t)}^{\gamma}$` | • <b>gamma (hetero_power)</b>: 1.0<br>• <b>max/min scale ratio</b>: 2.78 | Kiểm tra khả năng co giãn độ rộng khoảng phân vị theo tháng. |
| **S3: Base + Heavy-tailed Innovations (Student-t)** | `$\text{Base},\ \varepsilon_t\sim t_\nu\sqrt{(\nu-2)/\nu}$` | • <b>nu (t_df)</b>: 3.0 | Kiểm tra độ vững của phân vị đuôi (P5/P95) và chi phí tồn kho khi nhiễu đuôi dày. |
| **S4: Base + Level Shift in Test Period** | `$\text{Base},\ m_t\mathrel{+}=\delta\,\mathbb 1[t\ge\tau]$` | • <b>|delta| / mean</b>: (0.15, 0.25)<br>• <b>tau / n</b>: (0.8, 0.9)<br>• <b>sign</b>: ngẫu nhiên ± | Concept drift xảy ra trong giai đoạn test; đo khả năng thích ứng (vd. RevIN) khi mức nhu cầu đổi đột ngột. |
| **S5: Base + Trend Break** | `$\text{Base},\ m_t\mathrel{+}=\Delta_b\max(0,t-\tau)$` | • <b>|Delta_b| / mean per month</b>: (0.004, 0.008)<br>• <b>tau / n</b>: (0.7, 0.8)<br>• <b>sign</b>: ngẫu nhiên ± | Gãy trend quanh ranh giới train/val: mô hình phải ngoại suy độ dốc mới. |
| **S6: Base + Lagged Exogenous Driver (Promotion)** | `$\text{Base},\ m_t\mathrel{+}=\beta k\sigma_t\,\tilde g(z_{t-\ell}),\ \tilde g=\text{std}(\tanh z)$` | • <b>beta (exog_beta)</b>: 1.0<br>• <b>lag</b>: 1<br>• <b>driver</b>: PromotionAmount khử trend tuyến tính, chuẩn hóa median/IQR<br>• <b>share of innovation variance</b>: 0.5 | Đo lợi ích của mixing đa biến: chỉ mô hình nhìn thấy biến ngoại sinh trễ mới khai thác được phần tín hiệu này. |

---

## 2. Bảng Siêu Tham Số Tối Ưu Bền Vững Tìm Được Qua Optuna HPO (Trung bình trên Tuning Seeds [1000, 1001, 1002])

| Kịch bản | Mô hình | Lookback | Batch Size | Learning Rate | Hidden Size | N-Layers | N-Stacks/Blocks | Mean Val Pinball | Best Trial |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| S1 | **TSMIXER** | 18 | 8 | 3.69e-03 | 192 | 2 | - | 0.0226 | #11 |
| S1 | **NBEATS** | 9 | 16 | 1.75e-03 | 384 | 3 | 2/2 | 0.0058 | #27 |
| S1 | **NHITS** | 9 | 8 | 4.92e-03 | 192 | 3 | 3/3 | 0.0057 | #22 |
| S2 | **TSMIXER** | 18 | 8 | 1.04e-03 | 64 | 2 | - | 0.0244 | #7 |
| S2 | **NBEATS** | 12 | 8 | 2.42e-03 | 320 | 2 | 2/2 | 0.0053 | #42 |
| S2 | **NHITS** | 12 | 16 | 1.18e-03 | 512 | 4 | 2/3 | 0.0058 | #38 |
| S3 | **TSMIXER** | 18 | 8 | 1.04e-03 | 64 | 2 | - | 0.0196 | #7 |
| S3 | **NBEATS** | 9 | 32 | 4.44e-03 | 128 | 4 | 3/3 | 0.0043 | #45 |
| S3 | **NHITS** | 12 | 8 | 4.56e-04 | 192 | 4 | 2/2 | 0.0043 | #44 |
| S4 | **TSMIXER** | 18 | 8 | 3.69e-03 | 192 | 2 | - | 0.0226 | #11 |
| S4 | **NBEATS** | 9 | 16 | 1.75e-03 | 384 | 3 | 2/2 | 0.0058 | #27 |
| S4 | **NHITS** | 9 | 8 | 4.92e-03 | 192 | 3 | 3/3 | 0.0057 | #22 |
| S5 | **TSMIXER** | 18 | 8 | 3.69e-03 | 192 | 2 | - | 0.0225 | #11 |
| S5 | **NBEATS** | 9 | 16 | 2.91e-03 | 256 | 3 | 2/1 | 0.0060 | #21 |
| S5 | **NHITS** | 9 | 16 | 3.14e-03 | 192 | 3 | 2/3 | 0.0058 | #32 |
| S6 | **TSMIXER** | 18 | 8 | 1.52e-03 | 256 | 3 | - | 0.0209 | #24 |
| S6 | **NBEATS** | 9 | 16 | 6.63e-04 | 448 | 4 | 2/1 | 0.0050 | #26 |
| S6 | **NHITS** | 9 | 16 | 4.59e-03 | 192 | 3 | 3/2 | 0.0056 | #11 |

---

## 3. Bảng Tổng Hợp Skill Scores So Với Oracle (Trung Bình kèm Khoảng Tin Cậy 95% qua 30 Seeds)

> **Ý nghĩa chỉ số Skill Score so với Oracle:**  
> - **Pinball Loss Gap ($\Delta L = \text{Loss}_{\text{model}} - \text{Loss}_{\text{oracle}}$)**: Khoảng cách tiệm cận đến cận dưới lý thuyết hoàn hảo của phân phối.  
> - **Relative Pinball Efficiency ($RPE = \frac{\text{Loss}_{\text{oracle}}}{\text{Loss}_{\text{model}}} \times 100\%$)**: Tỷ lệ phần trăm hiệu suất đạt được so với Oracle (100% = ngang ngửa Oracle).  
> - **Cost Gap ($\Delta TC = \text{TC}_{\text{model}} - \text{TC}_{\text{oracle}}$)**: Chi phí tổn thất do bất định so với Oracle.

| Kịch bản | Mô hình | Test Pinball Loss (Mean ± 95% CI) | Pinball Loss Gap ($\Delta L$) | Relative Pinball Efficiency (%) | Chi Phí Tồn Kho (VND) (Mean ± 95% CI) | Cost Gap ($\Delta TC$) | Relative Cost Efficiency (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| S1 | ORACLE (DGP) | 1,056.3 [997.3, 1,115.4] | 0.0 | 100.00% | 841,945,954 [787,432,483, 896,459,425] | 0 | 100.00% |
| S1 | **TSMIXER** | 3,360.0 [3,083.9, 3,636.0] | +2,303.7 | 32.50% | 2,252,118,991 [2,038,538,759, 2,465,699,223] | +1,410,173,037 | 39.07% |
| S1 | **NBEATS** | 2,833.0 [2,595.7, 3,070.3] | +1,776.7 | 38.63% | 1,974,250,781 [1,827,665,107, 2,120,836,454] | +1,132,304,826 | 44.03% |
| S1 | **NHITS** | 3,018.9 [2,743.2, 3,294.6] | +1,962.6 | 36.70% | 2,151,404,815 [1,962,776,951, 2,340,032,679] | +1,309,458,861 | 41.30% |
| S2 | ORACLE (DGP) | 1,011.8 [952.8, 1,070.8] | 0.0 | 100.00% | 871,462,680 [808,681,403, 934,243,958] | 0 | 100.00% |
| S2 | **TSMIXER** | 3,286.6 [3,074.2, 3,499.1] | +2,274.8 | 31.59% | 2,178,366,127 [2,022,603,332, 2,334,128,922] | +1,306,903,447 | 41.24% |
| S2 | **NBEATS** | 2,846.9 [2,556.2, 3,137.6] | +1,835.1 | 37.45% | 2,021,227,276 [1,803,846,950, 2,238,607,603] | +1,149,764,596 | 45.49% |
| S2 | **NHITS** | 2,787.0 [2,554.5, 3,019.6] | +1,775.2 | 37.32% | 1,858,637,044 [1,708,520,184, 2,008,753,904] | +987,174,364 | 47.99% |
| S3 | ORACLE (DGP) | 891.5 [826.1, 957.0] | 0.0 | 100.00% | 658,152,667 [592,003,757, 724,301,576] | 0 | 100.00% |
| S3 | **TSMIXER** | 3,005.0 [2,745.6, 3,264.5] | +2,113.5 | 30.71% | 1,979,955,428 [1,745,535,879, 2,214,374,977] | +1,321,802,761 | 35.27% |
| S3 | **NBEATS** | 2,708.6 [2,408.1, 3,009.2] | +1,817.1 | 34.52% | 1,829,391,235 [1,630,004,717, 2,028,777,753] | +1,171,238,569 | 38.15% |
| S3 | **NHITS** | 2,628.9 [2,367.8, 2,890.0] | +1,737.4 | 36.03% | 1,733,468,704 [1,536,689,322, 1,930,248,086] | +1,075,316,038 | 40.85% |
| S4 | ORACLE (DGP) | 1,056.3 [997.3, 1,115.4] | 0.0 | 100.00% | 841,945,954 [787,432,483, 896,459,425] | 0 | 100.00% |
| S4 | **TSMIXER** | 12,381.6 [11,345.3, 13,417.8] | +11,325.3 | 9.08% | 6,609,128,171 [6,107,664,709, 7,110,591,633] | +5,767,182,217 | 13.40% |
| S4 | **NBEATS** | 9,720.8 [8,909.0, 10,532.6] | +8,664.5 | 11.60% | 6,881,119,329 [6,224,759,726, 7,537,478,933] | +6,039,173,375 | 13.03% |
| S4 | **NHITS** | 9,984.8 [9,114.0, 10,855.7] | +8,928.5 | 11.29% | 7,089,918,290 [6,436,093,697, 7,743,742,883] | +6,247,972,336 | 12.68% |
| S5 | ORACLE (DGP) | 1,056.3 [997.3, 1,115.4] | 0.0 | 100.00% | 841,945,954 [787,432,483, 896,459,425] | 0 | 100.00% |
| S5 | **TSMIXER** | 5,790.8 [5,189.8, 6,391.7] | +4,734.5 | 19.73% | 2,893,696,192 [2,565,337,621, 3,222,054,763] | +2,051,750,237 | 31.88% |
| S5 | **NBEATS** | 4,101.0 [3,718.2, 4,483.7] | +3,044.7 | 27.29% | 2,584,987,595 [2,347,865,178, 2,822,110,012] | +1,743,041,641 | 35.14% |
| S5 | **NHITS** | 4,021.9 [3,658.0, 4,385.8] | +2,965.5 | 27.71% | 2,560,492,894 [2,336,583,031, 2,784,402,758] | +1,718,546,940 | 34.73% |
| S6 | ORACLE (DGP) | 1,056.3 [997.3, 1,115.4] | 0.0 | 100.00% | 841,945,954 [787,432,483, 896,459,425] | 0 | 100.00% |
| S6 | **TSMIXER** | 3,545.3 [3,273.3, 3,817.4] | +2,489.0 | 31.03% | 2,310,509,626 [2,140,567,557, 2,480,451,695] | +1,468,563,672 | 37.64% |
| S6 | **NBEATS** | 3,478.0 [3,167.1, 3,789.0] | +2,421.7 | 31.82% | 2,232,470,802 [2,012,533,615, 2,452,407,990] | +1,390,524,848 | 40.12% |
| S6 | **NHITS** | 3,591.6 [3,346.7, 3,836.4] | +2,535.3 | 30.26% | 2,522,986,698 [2,297,680,774, 2,748,292,622] | +1,681,040,744 | 35.16% |

---

## 4. Bảng Kiểm Định Giả Thuyết Thống Kê So Sánh Bắt Cặp (Wilcoxon Signed-Rank Test qua 30 Seeds)

> **Quy ước mức ý nghĩa:**  
> - Giả thuyết $H_0$: Không có sự khác biệt có ý nghĩa giữa 2 mô hình trên cùng realization dữ liệu.  
> - Ký hiệu: `*** p < 0.001`, `** p < 0.01`, `* p < 0.05`, `ns: p >= 0.05 (Chấp nhận H0)`.  
> - **Tỷ lệ thắng (Win Rate %)**: Tỷ lệ phần trăm các seed mà Model 1 đạt loss thấp hơn Model 2.

| Kịch bản | Cặp So Sánh | Chỉ Số | Chênh Lệch Trung Bình (Mean Diff) | Tỷ Lệ Thắng (%) | Wilcoxon Statistic ($W$) | Wilcoxon $p$-value | Kết Luận Thống Kê |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| S1 | **TSMIXER vs NBEATS** | Pinball Loss | 527.0 | 23.3% | 98.0 | 0.0047 (**) | Khác biệt có ý nghĩa |
| S1 | **TSMIXER vs NBEATS** | Total Cost | 277,868,210.3 | 30.0% | 125.0 | 0.0262 (*) | Khác biệt có ý nghĩa |
| S1 | **TSMIXER vs NHITS** | Pinball Loss | 341.1 | 40.0% | 150.0 | 0.0919 (ns) | Tương đương (Chấp nhận $H_0$) |
| S1 | **TSMIXER vs NHITS** | Total Cost | 100,714,176.0 | 46.7% | 209.0 | 0.6408 (ns) | Tương đương (Chấp nhận $H_0$) |
| S1 | **NBEATS vs NHITS** | Pinball Loss | -185.9 | 60.0% | 149.0 | 0.0879 (ns) | Tương đương (Chấp nhận $H_0$) |
| S1 | **NBEATS vs NHITS** | Total Cost | -177,154,034.3 | 66.7% | 120.0 | 0.0197 (*) | Khác biệt có ý nghĩa |
| S2 | **TSMIXER vs NBEATS** | Pinball Loss | 439.7 | 23.3% | 95.0 | 0.0037 (**) | Khác biệt có ý nghĩa |
| S2 | **TSMIXER vs NBEATS** | Total Cost | 157,138,850.6 | 40.0% | 165.0 | 0.1706 (ns) | Tương đương (Chấp nhận $H_0$) |
| S2 | **TSMIXER vs NHITS** | Pinball Loss | 499.6 | 20.0% | 65.0 | 2.83e-04 (***) | Khác biệt có ý nghĩa |
| S2 | **TSMIXER vs NHITS** | Total Cost | 319,729,083.1 | 23.3% | 61.0 | 1.89e-04 (***) | Khác biệt có ý nghĩa |
| S2 | **NBEATS vs NHITS** | Pinball Loss | 59.9 | 46.7% | 211.0 | 0.6702 (ns) | Tương đương (Chấp nhận $H_0$) |
| S2 | **NBEATS vs NHITS** | Total Cost | 162,590,232.4 | 33.3% | 150.0 | 0.0919 (ns) | Tương đương (Chấp nhận $H_0$) |
| S3 | **TSMIXER vs NBEATS** | Pinball Loss | 296.4 | 33.3% | 132.0 | 0.0384 (*) | Khác biệt có ý nghĩa |
| S3 | **TSMIXER vs NBEATS** | Total Cost | 150,564,192.5 | 43.3% | 177.0 | 0.2621 (ns) | Tương đương (Chấp nhận $H_0$) |
| S3 | **TSMIXER vs NHITS** | Pinball Loss | 376.1 | 30.0% | 104.0 | 0.0071 (**) | Khác biệt có ý nghĩa |
| S3 | **TSMIXER vs NHITS** | Total Cost | 246,486,723.8 | 33.3% | 130.0 | 0.0345 (*) | Khác biệt có ý nghĩa |
| S3 | **NBEATS vs NHITS** | Pinball Loss | 79.7 | 53.3% | 222.0 | 0.8394 (ns) | Tương đương (Chấp nhận $H_0$) |
| S3 | **NBEATS vs NHITS** | Total Cost | 95,922,531.4 | 46.7% | 188.0 | 0.3707 (ns) | Tương đương (Chấp nhận $H_0$) |
| S4 | **TSMIXER vs NBEATS** | Pinball Loss | 2,660.8 | 6.7% | 22.0 | 9.98e-07 (***) | Khác biệt có ý nghĩa |
| S4 | **TSMIXER vs NBEATS** | Total Cost | -271,991,158.2 | 50.0% | 227.0 | 0.9193 (ns) | Tương đương (Chấp nhận $H_0$) |
| S4 | **TSMIXER vs NHITS** | Pinball Loss | 2,396.7 | 16.7% | 42.0 | 2.08e-05 (***) | Khác biệt có ý nghĩa |
| S4 | **TSMIXER vs NHITS** | Total Cost | -480,790,118.8 | 50.0% | 194.0 | 0.4400 (ns) | Tương đương (Chấp nhận $H_0$) |
| S4 | **NBEATS vs NHITS** | Pinball Loss | -264.1 | 56.7% | 183.0 | 0.3184 (ns) | Tương đương (Chấp nhận $H_0$) |
| S4 | **NBEATS vs NHITS** | Total Cost | -208,798,960.6 | 56.7% | 180.0 | 0.2894 (ns) | Tương đương (Chấp nhận $H_0$) |
| S5 | **TSMIXER vs NBEATS** | Pinball Loss | 1,689.8 | 16.7% | 21.0 | 8.33e-07 (***) | Khác biệt có ý nghĩa |
| S5 | **TSMIXER vs NBEATS** | Total Cost | 308,708,596.8 | 30.0% | 135.0 | 0.0449 (*) | Khác biệt có ý nghĩa |
| S5 | **TSMIXER vs NHITS** | Pinball Loss | 1,768.9 | 6.7% | 4.0 | 1.30e-08 (***) | Khác biệt có ý nghĩa |
| S5 | **TSMIXER vs NHITS** | Total Cost | 333,203,297.6 | 33.3% | 132.0 | 0.0384 (*) | Khác biệt có ý nghĩa |
| S5 | **NBEATS vs NHITS** | Pinball Loss | 79.1 | 43.3% | 201.0 | 0.5291 (ns) | Tương đương (Chấp nhận $H_0$) |
| S5 | **NBEATS vs NHITS** | Total Cost | 24,494,700.8 | 46.7% | 219.0 | 0.7922 (ns) | Tương đương (Chấp nhận $H_0$) |
| S6 | **TSMIXER vs NBEATS** | Pinball Loss | 67.3 | 43.3% | 196.0 | 0.4645 (ns) | Tương đương (Chấp nhận $H_0$) |
| S6 | **TSMIXER vs NBEATS** | Total Cost | 78,038,823.8 | 50.0% | 209.0 | 0.6408 (ns) | Tương đương (Chấp nhận $H_0$) |
| S6 | **TSMIXER vs NHITS** | Pinball Loss | -46.2 | 50.0% | 220.0 | 0.8078 (ns) | Tương đương (Chấp nhận $H_0$) |
| S6 | **TSMIXER vs NHITS** | Total Cost | -212,477,071.8 | 56.7% | 159.0 | 0.1347 (ns) | Tương đương (Chấp nhận $H_0$) |
| S6 | **NBEATS vs NHITS** | Pinball Loss | -113.5 | 66.7% | 157.0 | 0.1241 (ns) | Tương đương (Chấp nhận $H_0$) |
| S6 | **NBEATS vs NHITS** | Total Cost | -290,515,895.6 | 63.3% | 128.0 | 0.0310 (*) | Khác biệt có ý nghĩa |

---

## 5. Tóm Tắt Kết Quả (sinh tự động từ số liệu, cần tác giả diễn giải thêm)

1. **S1** — Thứ hạng Pinball: NBEATS (2,833, RPE 38.6%) < NHITS (3,019, RPE 36.7%) < TSMIXER (3,360, RPE 32.5%). Mô hình tốt nhất: **NBEATS**.  
   Khác biệt có ý nghĩa (Wilcoxon, p<0.05): NBEATS < TSMIXER (0.0047 (**)).
2. **S2** — Thứ hạng Pinball: NHITS (2,787, RPE 37.3%) < NBEATS (2,847, RPE 37.4%) < TSMIXER (3,287, RPE 31.6%). Mô hình tốt nhất: **NHITS**.  
   Khác biệt có ý nghĩa (Wilcoxon, p<0.05): NBEATS < TSMIXER (0.0037 (**)), NHITS < TSMIXER (2.83e-04 (***)).
3. **S3** — Thứ hạng Pinball: NHITS (2,629, RPE 36.0%) < NBEATS (2,709, RPE 34.5%) < TSMIXER (3,005, RPE 30.7%). Mô hình tốt nhất: **NHITS**.  
   Khác biệt có ý nghĩa (Wilcoxon, p<0.05): NBEATS < TSMIXER (0.0384 (*)), NHITS < TSMIXER (0.0071 (**)).
4. **S4** — Thứ hạng Pinball: NBEATS (9,721, RPE 11.6%) < NHITS (9,985, RPE 11.3%) < TSMIXER (12,382, RPE 9.1%). Mô hình tốt nhất: **NBEATS**.  
   Khác biệt có ý nghĩa (Wilcoxon, p<0.05): NBEATS < TSMIXER (9.98e-07 (***)), NHITS < TSMIXER (2.08e-05 (***)).
5. **S5** — Thứ hạng Pinball: NHITS (4,022, RPE 27.7%) < NBEATS (4,101, RPE 27.3%) < TSMIXER (5,791, RPE 19.7%). Mô hình tốt nhất: **NHITS**.  
   Khác biệt có ý nghĩa (Wilcoxon, p<0.05): NBEATS < TSMIXER (8.33e-07 (***)), NHITS < TSMIXER (1.30e-08 (***)).
6. **S6** — Thứ hạng Pinball: NBEATS (3,478, RPE 31.8%) < TSMIXER (3,545, RPE 31.0%) < NHITS (3,592, RPE 30.3%). Mô hình tốt nhất: **NBEATS**.  
   Khác biệt có ý nghĩa (Wilcoxon, p<0.05): không có.

---
*Báo cáo được khởi tạo tự động bởi `run_simulation_benchmark.py` phục vụ công bố khoa học.*