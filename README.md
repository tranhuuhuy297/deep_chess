# CHESS
*Paper: https://arxiv.org/pdf/1711.09667v1.pdf
*Khuyên dùng: Ubuntu 18.04 +
*Cài đặt môi trường:
```
pip install -r requirements.txt
```
## Play Chess
```
python play.py
```
## Train
Xử  lý dữ liệu từ các game records
```
python parse_data.py
```
Train model Auto Encoder 
```
python train_ae.py
```
Lấy output của 2 nhánh model Auto Encoder làm input cho model Siamese
```
python featurize.py
```
Train model Siamese
```
python train_siamese.py
```
