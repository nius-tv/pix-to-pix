mkdir /data
mkdir /models

gsutil \
    -m copy \
    -r gs://plasmic-training/prep-video-train/* \
    /data

docker build \
    -t pix-to-pix \
    .

gcloud docker -- pull us.gcr.io/plasmic-artefacts-2/pix-to-pix

# Inference
docker run \
    -v $(PWD)/data/speech-to-landmarks/inference:/data/landmarks \
    -v $(PWD)/data/inference:/data/inferred \
    -v $(PWD)/models:/models \
    -v $(PWD)/pix-to-pix:/app \
    -it us.gcr.io/plasmic/pix-to-pix \
# Train
nvidia-docker run \
    --shm-size 2G \
    -v /data:/data \
    -v /models:/models \
    -it us.gcr.io/plasmic-artefacts-2/pix-to-pix \
    bash

# TODO: experiment with different settings in "train.py". 
See https://github.com/NVIDIA/pix2pixHD/blob/5a2c87201c5957e2bf51d79b8acddb9cc1920b26/options/base_options.py#L13-L61

# --continue_train \
python train.py \
    --checkpoints_dir /models \
    --dataroot /data \
    --label_nc 0 \
    --name nius \
    --no_flip \
    --no_instance \
    --resize_or_crop none \
    --verbose
