docker build \
    -t nvidia-pix2pixHD \
    .

TODO: experiment with different settings in "train.py". See https://github.com/NVIDIA/pix2pixHD/blob/5a2c87201c5957e2bf51d79b8acddb9cc1920b26/options/base_options.py#L13-L61

python train.py \
    --checkpoints_dir /models/pix2pix/
    --continue_train \
    --dataroot /data/pix2pix/ \
    --label_nc 0 \
    --name nius \
    --no_flip \
    --no_instance \
    --resize_or_crop none \
    --verbose
