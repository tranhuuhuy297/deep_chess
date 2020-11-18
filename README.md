# CHESS
* Paper: https://arxiv.org/pdf/1711.09667v1.pdf
* Khuyên dùng: Ubuntu 18.04+, cuda 10.1, python 3.6+
* Cài đặt môi trường:
```
pip install -r requirements.txt
```
* Ubuntu (with cuda)
```
pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
```
## Play Chess
```
python play.py
```
## Train
* Xử  lý dữ liệu từ các game records
```
python parse_data.py
```
* Train model Auto Encoder 
```
python train_ae.py
```
* Lấy output của 2 nhánh model Auto Encoder làm input cho model Siamese
```
python featurize.py
```
* Train model Siamese
```
python train_siamese.py
```
