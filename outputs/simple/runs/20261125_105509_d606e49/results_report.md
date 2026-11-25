# SLR — Results and discussion

**Dataset and setup.** 50,000 retained samples from existing CSV partitions; train/validation/test = 35,000/7,500/7,500; refit = 42,500. Train GPA at 4: 3,369 (9.63%).

**M01 — GenAI hours.** Final slope -0.001351 GPA per additional weekly hour (unadjusted association). Raw test MAE 0.40645, RMSE 0.49754, R² -0.00069; clipped MAE 0.40645, RMSE 0.49754, R² -0.00069. Raw out-of-range count: 0.

**M03 — prior GPA.** Raw RMSE 0.18509, R² 0.86152; clipped RMSE 0.18395, R² 0.86322. Raw above 4: 350. RMSE improvement: 0.616%.

**Validation selection.** No clipping: M03; clipping: M03. Mean baseline test RMSE 0.49737.

**Conclusion.** Hệ số giống nhau; clipping đảm bảo đầu ra hợp lệ, chỉ thay đổi dự đoán ngoài miền. Metric tốt hơn (nếu có) do projection, không phải học hệ số tốt hơn. M01 phản ánh liên hệ đơn biến chứ không chứng minh tác động nhân quả của GenAI.

**Limitations and future work.** SLR không kiểm soát nhiễu/tương tác; nguồn và thời điểm ghi nhận feature chưa xác minh độc lập. Cách chia dữ liệu có sẵn chưa được xác minh độc lập và không chứng minh tổng quát hóa qua trường/kỳ học. Ceiling effect và ngoại suy cần được xem xét. Có thể nghiên cứu hồi quy đa biến hoặc bounded/censored regression trong thí nghiệm mới.
