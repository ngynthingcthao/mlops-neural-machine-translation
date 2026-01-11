# mlops-neural-machine-translation
## Giới thiệu
Dự án này xây dựng một **hệ thống dịch máy Neural Machine Translation (NMT) cho cặp ngôn ngữ tiếng Anh – tiếng Việt** theo hướng **MLOps**. Hệ thống hỗ trợ huấn luyện mô hình **Transformer từ đầu** và **fine-tune mô hình pretrained OPUS**, đồng thời tự động hóa toàn bộ quy trình từ tiền xử lý dữ liệu, huấn luyện, đánh giá đến triển khai mô hình thông qua **DVC**, **Weights & Biases** và **FastAPI**.  
Mục tiêu của đề tài là đảm bảo **tính tái lập**, **khả năng quản lý pipeline**, và **dễ dàng triển khai mô hình dịch máy trong thực tế**.
# Cách chạy
dvc repro
mlflow ui --port 5000
uvicorn serving.app:app --reload --port 8000
