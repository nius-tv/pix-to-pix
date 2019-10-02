FROM floydhub/pytorch:0.3.0-gpu.cuda8cudnn6-py3.24

RUN apt-get update -y

RUN pip install dominate==2.4.0

COPY . /app

ENV PYTHONPATH=/app
ENV PYTHONPATH=/app/nvidia-pix2pixHD:$PYTHONPATH

# Install gcsfuse
RUN apt-get install -y curl
RUN apt-get install -y lsb-release

RUN export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s` && \
	echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | \
	tee /etc/apt/sources.list.d/gcsfuse.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

RUN apt-get update -y
RUN apt-get install -y gcsfuse

# Install gcloud
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | \
	tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

RUN apt-get install -y apt-transport-https
RUN apt-get install -y ca-certificates

RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
	apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

RUN apt-get update -y
RUN apt-get install -y google-cloud-sdk
