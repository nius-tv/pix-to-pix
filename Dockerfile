FROM floydhub/pytorch:0.3.0-gpu.cuda8cudnn6-py3.24

RUN apt-get update -y

RUN pip install dominate==2.4.0

COPY . /app
