FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN apt-get update -y

RUN apt-get install -y python3.5
RUN apt-get install -y python3-pip

RUN pip3 install dominate==2.4.0
RUN pip3 install image==1.5.27
RUN pip3 install scipy==1.3.2
RUN pip3 install torch==1.3.1
RUN pip3 install torchvision==0.4.2

RUN apt-get install -y git
RUN pip3 install --upgrade pip
RUN git clone https://github.com/NVIDIA/apex && \
	cd apex && \
	python3 -m pip install \
		-v \
		--no-cache-dir \
		--global-option="--cpp_ext" \
		--global-option="--cuda_ext" \
		.

COPY . /app
