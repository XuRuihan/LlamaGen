# !/bin/bash
set -x

torchrun \
    --nnodes=1 --nproc_per_node=4 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port=12345 \
    tokenizer/tokenizer_image/vq_train_with_eval.py --cloud-save-path ./outputs --train-data-path /disk2/xrh/datasets/imagenet1k/train --valid-data-path /disk2/xrh/datasets/imagenet1k/val \
    --image-size 128 --image-size-eval 128 --vq-model VQ-16 --global-batch-size 64
