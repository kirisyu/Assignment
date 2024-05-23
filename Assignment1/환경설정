# Assignment1

python 3.11
tensorflow 2.6

dockerfile내용
FROM tensorflow/tensorflow:latest
WORKDIR /app
COPY best_model.h5 /app/best_model.h5
COPY serve_model.py /app/serve_model.py
RUN pip install flask
CMD ["python", "serve_model.py"]

curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"input": [[CIRAR-10]]}'
CIRAR-10 : CIRAR-10 이미지 데이터 포함된 배열을 입력하면 동작
