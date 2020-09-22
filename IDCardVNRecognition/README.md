# IDCardVNRecognition

*Recommend: Ubuntu, cuda10.1

Run tensorflow_model_server:

sudo apt install tesseract-ocr
sudo apt install libtesseract-dev
sudo apt install tesseract-ocr-vie
! pip install --quiet vietocr==0.1.9

# step 1
$ echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -

# step 2
$ sudo apt-get update && sudo apt-get install tensorflow-model-server

# step 3
$ pip install tensorflow-serving-api

# step 4
$ tensorflow_model_server --port=8500 --rest_api_port=8501 --model_config_file=./models/serving.config

# step 5
./app/static/images: folder trong server chứa ảnh được upload lên

./models/serving.config: chỉnh lại base_path của bạn ( "/home/{user_name}/.../IDCardRecognition/models/{model_name}"

# step 6
$ python run.py


#  install tourch
pip install --no-cache-dir torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html 